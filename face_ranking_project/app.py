"""
人脸偏好打分排序 Web 工具 - Flask 主程序

启动后自动扫描 static/faces 目录，提供成对对比与排名展示。
所有数据本地存储，无网络上传。
"""

from __future__ import annotations

import csv
import io
import socket
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from ranking_engine import RankingEngine

app = Flask(__name__)
engine = RankingEngine()


def _is_port_available(host: str, port: int) -> bool:
    """检查本机端口是否可用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _resolve_runtime_port() -> int:
    """
    解析最终运行端口：
    - 默认尝试 config.FLASK_PORT
    - 若占用且开启自动回退，则尝试后续端口
    """
    host = config.FLASK_HOST
    requested = int(config.FLASK_PORT)
    if _is_port_available(host, requested):
        return requested

    if not config.FLASK_AUTO_PORT_FALLBACK:
        raise RuntimeError(f"端口 {requested} 已被占用")

    search_limit = max(1, int(config.FLASK_PORT_SEARCH_LIMIT))
    for candidate in range(requested + 1, requested + search_limit + 1):
        if _is_port_available(host, candidate):
            print(f"[提示] 端口 {requested} 被占用，自动切换至 {candidate}")
            return candidate

    raise RuntimeError(
        f"端口 {requested} 被占用，且在 +{search_limit} 范围内未找到可用端口"
    )


def scan_face_images() -> list[str]:
    """
    扫描 static/faces 下所有有效图片文件名。
    跳过无法识别的文件，避免程序崩溃。
    """
    faces_dir = config.FACES_DIR
    faces_dir.mkdir(parents=True, exist_ok=True)

    found: list[str] = []
    if not faces_dir.is_dir():
        print(f"[警告] 人脸目录不存在: {faces_dir}")
        return found

    for path in sorted(faces_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in config.ALLOWED_IMAGE_EXTENSIONS:
            continue
        try:
            # 尝试打开文件头，过滤损坏图片
            with path.open("rb") as f:
                header = f.read(16)
            if len(header) < 4:
                print(f"[跳过] 文件过小或损坏: {path.name}")
                continue
            found.append(path.name)
        except OSError as exc:
            print(f"[跳过] 无法读取图片 {path.name}: {exc}")

    return found


def refresh_faces() -> list[str]:
    """扫描磁盘并同步到排名引擎。"""
    face_ids = scan_face_images()
    engine.sync_faces(face_ids)
    return face_ids


@app.route("/")
def index():
    """成对对比主页。"""
    return render_template("index.html")


@app.route("/results")
def results_page():
    """排名结果页。"""
    return render_template("results.html")


@app.route("/faces/<path:filename>")
def serve_face(filename: str):
    """安全地提供本地人脸图片。"""
    safe_name = Path(filename).name
    faces_dir = config.FACES_DIR
    target = faces_dir / safe_name
    if not target.is_file():
        return jsonify({"error": "图片不存在"}), 404
    try:
        return send_from_directory(faces_dir, safe_name)
    except OSError as exc:
        return jsonify({"error": f"读取图片失败: {exc}"}), 500


@app.route("/api/status")
def api_status():
    refresh_faces()
    return jsonify(engine.get_status())


@app.route("/api/pair")
def api_pair():
    refresh_faces()
    status = engine.get_status()

    if status["total_faces"] < 2:
        return jsonify(
            {
                "ok": False,
                "error": "至少需要 2 张人脸图片，请将图片放入 static/faces 目录",
                "status": status,
            }
        )

    engine.ensure_comparison_active()
    status = engine.get_status()

    pair = engine.select_next_pair()
    if pair is None:
        return jsonify(
            {
                "ok": False,
                "error": "无法生成新的对比组",
                "status": status,
            }
        )

    left_id, right_id = pair
    return jsonify(
        {
            "ok": True,
            "left": {"id": left_id, "url": f"/faces/{left_id}"},
            "right": {"id": right_id, "url": f"/faces/{right_id}"},
            "status": status,
        }
    )


@app.route("/api/vote", methods=["POST"])
def api_vote():
    refresh_faces()
    engine.ensure_comparison_active()
    body = request.get_json(silent=True) or {}

    left_id = str(body.get("left_id", "")).strip()
    right_id = str(body.get("right_id", "")).strip()
    choice = str(body.get("choice", "")).strip().lower()

    if not left_id or not right_id:
        return jsonify({"ok": False, "error": "缺少图片 ID"}), 400

    result = engine.record_vote(left_id, right_id, choice)
    if not result.get("ok"):
        return jsonify(result), 400

    result["status"] = engine.get_status()
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    refresh_faces()
    engine.reset_all()
    return jsonify({"ok": True, "status": engine.get_status()})


@app.route("/api/rankings")
def api_rankings():
    refresh_faces()
    rankings = engine.compute_rankings()
    return jsonify(
        {
            "ok": True,
            "rankings": [
                {
                    **item,
                    "url": f"/faces/{item['id']}",
                }
                for item in rankings
            ],
            "status": engine.get_status(),
            "total_comparisons": engine.get_total_comparisons(),
        }
    )


@app.route("/api/export/csv")
def api_export_csv():
    refresh_faces()
    output = io.StringIO()
    writer = csv.writer(output)
    for row in engine.export_csv_rows():
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8-sig")
    return app.response_class(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="face_rankings.csv"'},
    )


def main() -> None:
    print("=" * 50)
    print("人脸偏好打分排序工具")
    print("=" * 50)

    faces = refresh_faces()
    print(f"扫描到 {len(faces)} 张人脸图片: {config.FACES_DIR}")
    if len(faces) < 2:
        print("[提示] 请向 static/faces 目录添加至少 2 张图片后再开始对比")

    print(f"最大对比轮次: {config.MAX_ITERATIONS}")
    runtime_port = _resolve_runtime_port()
    print(f"访问地址: http://{config.FLASK_HOST}:{runtime_port}")
    print("=" * 50)

    app.run(
        host=config.FLASK_HOST,
        port=runtime_port,
        debug=config.FLASK_DEBUG,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)
