"""
从 Wikimedia Commons 抓取真人照片到 static/faces。

用法示例：
python fetch_real_faces.py --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_QUERY = (
    'portrait photograph person "human face" -illustration -drawing '
    "-painting -anime -cartoon"
)
USER_AGENT = "face-ranking-project/1.0 (local research tool)"
CHUNK_SIZE = 1024 * 64

MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取真人照片到 static/faces")
    parser.add_argument("--limit", type=int, default=50, help="最大下载数量，默认 50")
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Wikimedia 搜索词，默认偏向真人肖像",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.FACES_DIR,
        help="下载目录，默认 static/faces",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="最小宽度，默认 256",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=256,
        help="最小高度，默认 256",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="每次下载间隔秒数，默认 0.2，避免请求过快",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="网络请求超时秒数，默认 20",
    )
    return parser.parse_args()


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, target: Path, timeout: float) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, target.open("wb") as f:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)


def _build_search_url(query: str, offset: int, batch_size: int = 50) -> str:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File namespace
        "gsrlimit": str(batch_size),
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
    }
    return f"{WIKIMEDIA_API}?{urllib.parse.urlencode(params)}"


def _iter_image_candidates(query: str, timeout: float):
    offset = 0
    while True:
        data = _http_get_json(_build_search_url(query, offset), timeout=timeout)
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            break

        for page in pages.values():
            image_infos = page.get("imageinfo") or []
            if not image_infos:
                continue
            info = image_infos[0]
            yield {
                "title": page.get("title", ""),
                "url": info.get("url", ""),
                "mime": info.get("mime", ""),
                "width": int(info.get("width", 0) or 0),
                "height": int(info.get("height", 0) or 0),
            }

        next_offset = (((data.get("continue") or {}).get("gsroffset")))
        if next_offset is None:
            break
        offset = int(next_offset)


def _next_file_index(output_dir: Path) -> int:
    max_index = 0
    for file in output_dir.iterdir():
        if not file.is_file():
            continue
        stem = file.stem
        if not stem.startswith("wm_face_"):
            continue
        suffix = stem.removeprefix("wm_face_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _is_allowed_candidate(item: dict, min_width: int, min_height: int) -> bool:
    if item["mime"] not in MIME_TO_EXT:
        return False
    if not item["url"]:
        return False
    if item["width"] < min_width or item["height"] < min_height:
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit 必须 > 0")
    if args.min_width <= 0 or args.min_height <= 0:
        raise ValueError("--min-width / --min-height 必须 > 0")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wikimedia 真人照片抓取器")
    print(f"保存目录: {output_dir}")
    print(f"目标数量: {args.limit}")
    print(f"搜索词: {args.query}")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    failed = 0
    index = _next_file_index(output_dir)
    seen_url_hash: set[str] = set()

    for item in _iter_image_candidates(args.query, timeout=args.timeout):
        if downloaded >= args.limit:
            break

        url = item["url"]
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()
        if url_hash in seen_url_hash:
            skipped += 1
            continue
        seen_url_hash.add(url_hash)

        if not _is_allowed_candidate(item, args.min_width, args.min_height):
            skipped += 1
            continue

        ext = MIME_TO_EXT[item["mime"]]
        filename = f"wm_face_{index:04d}{ext}"
        target = output_dir / filename
        index += 1

        try:
            _download_file(url, target, timeout=args.timeout)
            print(
                f"[OK] {filename} <- {item['title']} "
                f"({item['width']}x{item['height']})"
            )
            downloaded += 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failed += 1
            if target.exists():
                target.unlink(missing_ok=True)
            print(f"[失败] 下载失败 {filename}: {exc}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("=" * 60)
    print(f"完成，下载成功: {downloaded} 张")
    print(f"跳过: {skipped}，失败: {failed}")
    print(f"图片目录: {output_dir}")
    print("=" * 60)

    if downloaded == 0:
        print("[提示] 未下载到图片，可尝试更换 --query 或降低尺寸限制")


if __name__ == "__main__":
    main()
