"""
从百度图片搜索抓取照片到 static/faces（速度优先）。

默认关键词：女学生
示例页：https://image.baidu.com/search/index?tn=baiduimage&word=%E5%A5%B3%E5%AD%A6%E7%94%9F

仅保留：百度 acjson 接口 + 高并发下载 + 单行进度条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import config

BAIDU_ACJSON = "https://image.baidu.com/search/acjson"
DEFAULT_WORD = "女学生"
DEFAULT_REFERER = (
    "https://image.baidu.com/search/index?tn=baiduimage&fm=result&ie=utf-8&word="
    + urllib.parse.quote(DEFAULT_WORD)
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

CHUNK_SIZE = 1024 * 64
MIN_FILE_BYTES = 4 * 1024
MAX_FILE_BYTES = 15 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从百度图片抓取照片到 static/faces")
    parser.add_argument("--limit", type=int, default=200, help="目标成功数量，默认 200")
    parser.add_argument(
        "--source",
        choices=("baidu", "randomuser", "wikimedia", "auto"),
        default="baidu",
        help="兼容旧参数；randomuser/wikimedia/auto 已弃用，均使用百度图片",
    )
    parser.add_argument("--word", type=str, default=DEFAULT_WORD, help="搜索关键词，默认 女学生")
    parser.add_argument("--output-dir", type=Path, default=config.FACES_DIR)
    parser.add_argument("--workers", type=int, default=48, help="并发下载线程数，默认 48")
    parser.add_argument("--rn", type=int, default=60, help="每页抓取条数，默认 60")
    parser.add_argument("--max-pages", type=int, default=30, help="最多翻页数，默认 30")
    parser.add_argument("--timeout", type=float, default=8.0, help="单次请求超时秒数")
    return parser.parse_args()


def _render_progress(done: int, total: int, failed: int, skipped: int, width: int = 28) -> None:
    ratio = min(1.0, max(0.0, done / max(1, total)))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {done}/{total} (failed={failed}, skipped={skipped})",
        end="",
        flush=True,
    )


def _next_file_index(output_dir: Path) -> int:
    max_index = 0
    for file in output_dir.iterdir():
        if not file.is_file():
            continue
        stem = file.stem
        if not stem.startswith("bd_face_"):
            continue
        suffix = stem.removeprefix("bd_face_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _referer_for_word(word: str) -> str:
    return (
        "https://image.baidu.com/search/index?tn=baiduimage&fm=result&ie=utf-8&word="
        + urllib.parse.quote(word)
    )


def _parse_baidu_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # 百度偶发非法转义，做一次宽松清理
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = text.replace("\\'", "'")
    return json.loads(text)


def _build_list_url(word: str, pn: int, rn: int) -> str:
    params = {
        "tn": "resultjson_com",
        "ipn": "rj",
        "ct": "201326592",
        "fp": "result",
        "queryWord": word,
        "word": word,
        "cl": "2",
        "lm": "-1",
        "ie": "utf-8",
        "oe": "utf-8",
        "st": "-1",
        "ic": "0",
        "pn": str(pn),
        "rn": str(rn),
        "gsm": hex(pn)[2:],
    }
    return BAIDU_ACJSON + "?" + urllib.parse.urlencode(params)


def _pick_image_url(item: dict) -> str:
    # 缩略图更小，下载更快
    for key in ("thumbURL", "middleURL", "hoverURL", "objURL"):
        url = (item.get(key) or "").strip()
        if url.startswith("http://") or url.startswith("https://"):
            return url.replace("\\/", "/")
    return ""


def _fetch_baidu_page(word: str, pn: int, rn: int, timeout: float) -> list[str]:
    url = _build_list_url(word, pn, rn)
    headers = dict(HEADERS)
    headers["Referer"] = _referer_for_word(word)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = _parse_baidu_json(resp.read().decode("utf-8", errors="ignore"))

    urls: list[str] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        img = _pick_image_url(item)
        if img:
            urls.append(img)
    return urls


def _iter_baidu_urls(word: str, rn: int, max_pages: int, timeout: float):
    for page in range(max_pages):
        pn = page * rn
        try:
            urls = _fetch_baidu_page(word, pn, rn, timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            continue
        if not urls:
            break
        for u in urls:
            yield u


def _download_file(url: str, target: Path, referer: str, timeout: float) -> bool:
    headers = dict(HEADERS)
    headers["Referer"] = referer
    headers["Accept"] = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, target.open("wb") as f:
            total = 0
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError("too large")
                f.write(chunk)
        if target.stat().st_size < MIN_FILE_BYTES:
            target.unlink(missing_ok=True)
            return False
        return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        if target.exists():
            target.unlink(missing_ok=True)
        return False


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit 必须 > 0")
    if args.workers <= 0:
        raise ValueError("--workers 必须 > 0")
    if args.source != "baidu":
        print(
            f"[提示] --source {args.source} 已弃用，现统一使用百度图片（--word 指定关键词）"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    referer = _referer_for_word(args.word)

    downloaded = 0
    failed = 0
    skipped = 0
    index = _next_file_index(output_dir)
    seen: set[str] = set()

    _render_progress(0, args.limit, failed=0, skipped=0)

    url_iter = _iter_baidu_urls(args.word, args.rn, args.max_pages, args.timeout)
    pending: dict = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while downloaded < args.limit:
            # 维持下载队列，避免空转
            while len(pending) < args.workers * 2 and downloaded + len(pending) < args.limit:
                try:
                    url = next(url_iter)
                except StopIteration:
                    break

                key = hashlib.sha1(url.encode("utf-8")).hexdigest()
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)

                target = output_dir / f"bd_face_{index:04d}.jpg"
                index += 1
                fut = pool.submit(_download_file, url, target, referer, args.timeout)
                pending[fut] = target

            if not pending:
                break

            done_set, pending_set = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done_set:
                pending.pop(fut, None)
                ok = fut.result()
                if ok:
                    downloaded += 1
                else:
                    failed += 1
                _render_progress(downloaded, args.limit, failed=failed, skipped=skipped)
                if downloaded >= args.limit:
                    break

            pending = {f: t for f, t in pending.items() if f in pending_set}

    print()
    print(f"完成: 成功 {downloaded}，失败 {failed}，跳过 {skipped}，关键词「{args.word}」")
    print(f"目录: {output_dir}")


if __name__ == "__main__":
    main()
