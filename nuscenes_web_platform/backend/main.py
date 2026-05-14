"""FastAPI app: NuScenes read-only API + static SPA."""

from __future__ import annotations

import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

# demo_by_nuscenes utilities (get_sample_data_path, BEV helpers, …)
_WEB_ROOT = Path(__file__).resolve().parents[1]
_DEMO_ROOT = _WEB_ROOT.parent / "demo_by_nuscenes"
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

import mimetypes

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from nuscenes.nuscenes import NuScenes

from backend.config import get_settings
from backend.media_paths import resolve_under_dataroot
from backend.visuals import (
    list_camera_channels,
    render_annotation_bev_png,
    render_camera_with_boxes_png,
    render_lidar_bev_png,
)

from utils import load_nuscenes  # type: ignore[import-untyped]

# Browsers can cache aggressively: URLs map to fixed dataset files / rendered keys.
_IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


def _radar_channels(nusc: NuScenes) -> list[str]:
    """Discover RADAR channels without scanning the full sample_data table."""
    seen: set[str] = set()
    for i, s in enumerate(nusc.sample):
        if i > 800:
            break
        for ch in s["data"]:
            if ch.startswith("RADAR"):
                seen.add(ch)
    return sorted(seen)


class RenderLRU:
    """Tiny in-memory cache for rendered PNG bytes."""

    def __init__(self, max_items: int = 64) -> None:
        self._max = max_items
        self._data: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_dataset_layout()
    nusc = load_nuscenes(str(settings.nuscenes_dataroot), settings.nuscenes_version)
    app.state.nusc = nusc
    app.state.settings = settings
    app.state.render_cache = RenderLRU(512)
    # scene_token -> ordered sample tokens
    scene_samples: dict[str, list[str]] = {}
    for scene in nusc.scene:
        tok = scene["token"]
        chain: list[str] = []
        t = scene["first_sample_token"]
        while t:
            chain.append(t)
            t = nusc.get("sample", t)["next"]
        scene_samples[tok] = chain
    app.state.scene_samples = scene_samples
    app.state.search_cache = {}
    yield


app = FastAPI(title="NuScenes Local Web", lifespan=lifespan)

_STATIC = _WEB_ROOT / "static"
if _STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_STATIC)), name="assets")


def get_nusc(request: Request) -> NuScenes:
    return request.app.state.nusc


def get_render_cache(request: Request) -> RenderLRU:
    return request.app.state.render_cache


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta(request: Request, nusc: NuScenes = Depends(get_nusc)):
    cams = list_camera_channels()
    radars = _radar_channels(nusc)
    categories = sorted({c["name"] for c in nusc.category})
    return {
        "dataroot": str(nusc.dataroot),
        "version": nusc.version,
        "camera_channels": cams,
        "radar_channels": radars,
        "categories": categories,
        "lidar_channel": "LIDAR_TOP",
    }


@app.get("/api/scenes")
def list_scenes(nusc: NuScenes = Depends(get_nusc)):
    return [
        {
            "token": s["token"],
            "name": s["name"],
            "description": s.get("description", ""),
            "nbr_samples": s["nbr_samples"],
        }
        for s in nusc.scene
    ]


@app.get("/api/scenes/{scene_token}/ego_trail")
def ego_trail(scene_token: str, request: Request):
    """World-frame ego trajectory (x,y from ego_pose at each keyframe) for map-style plot."""
    nusc: NuScenes = request.app.state.nusc
    try:
        nusc.get("scene", scene_token)
    except KeyError:
        raise HTTPException(404, "Unknown scene") from None
    chain = request.app.state.scene_samples.get(scene_token, [])
    points: list[dict] = []
    for st in chain:
        s = nusc.get("sample", st)
        lid = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lid["ego_pose_token"])
        t = ego["translation"]
        points.append({"sample_token": st, "x": float(t[0]), "y": float(t[1])})
    return {"scene_token": scene_token, "points": points}


@app.get("/api/scenes/{scene_token}/samples")
def scene_samples(
    scene_token: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    nusc: NuScenes = request.app.state.nusc
    try:
        nusc.get("scene", scene_token)
    except KeyError:
        raise HTTPException(404, "Unknown scene") from None
    chain = request.app.state.scene_samples.get(scene_token, [])
    total = len(chain)
    start = (page - 1) * page_size
    slice_tokens = chain[start : start + page_size]
    rows = []
    for st in slice_tokens:
        s = nusc.get("sample", st)
        rows.append(
            {
                "token": st,
                "timestamp": s["timestamp"],
                "scene_token": s["scene_token"],
            }
        )
    return {"total": total, "page": page, "page_size": page_size, "items": rows}


@app.get("/api/samples/{sample_token}")
def sample_detail(sample_token: str, nusc: NuScenes = Depends(get_nusc)):
    try:
        s = nusc.get("sample", sample_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    data_keys = list(s["data"].keys())
    scene = nusc.get("scene", s["scene_token"])
    return {
        "token": s["token"],
        "timestamp": s["timestamp"],
        "scene_token": s["scene_token"],
        "scene_name": scene["name"],
        "data_channels": data_keys,
        "ann_count": len(s["anns"]),
    }


@app.get("/api/samples/{sample_token}/media")
def sample_media_url(
    sample_token: str,
    channel: str = Query(...),
    nusc: NuScenes = Depends(get_nusc),
):
    try:
        s = nusc.get("sample", sample_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    if channel not in s["data"]:
        raise HTTPException(400, f"Channel {channel!r} not in sample")
    sd_token = s["data"][channel]
    return {"sample_data_token": sd_token, "media_url": f"/api/media/sample_data/{sd_token}"}


@app.get("/api/samples/{sample_token}/raw_image/{channel}")
def sample_raw_image_file(
    sample_token: str,
    channel: str,
    nusc: NuScenes = Depends(get_nusc),
):
    """Direct image bytes for <img src> without a prior JSON round-trip."""
    try:
        s = nusc.get("sample", sample_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    if channel not in s["data"]:
        raise HTTPException(404, f"Channel {channel!r} not in sample")
    sd = nusc.get("sample_data", s["data"][channel])
    path = resolve_under_dataroot(Path(nusc.dataroot), sd["filename"])
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime or "image/jpeg", headers=_IMAGE_CACHE_HEADERS)


@app.get("/api/samples/{sample_token}/annotations")
def sample_annotations(sample_token: str, nusc: NuScenes = Depends(get_nusc)):
    try:
        s = nusc.get("sample", sample_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    items = []
    for ann_token in s["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        items.append(
            {
                "token": ann["token"],
                "category_name": ann["category_name"],
                "translation": ann["translation"],
                "size": ann["size"],
                "rotation": ann["rotation"],
                "velocity": ann.get("velocity", [0.0, 0.0]),
                "num_lidar_pts": ann.get("num_lidar_pts", -1),
                "num_radar_pts": ann.get("num_radar_pts", -1),
            }
        )
    return {"items": items}


@app.get("/api/samples/{sample_token}/ego")
def sample_ego(sample_token: str, nusc: NuScenes = Depends(get_nusc)):
    try:
        s = nusc.get("sample", sample_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    lidar_token = s["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    return {
        "token": ego["token"],
        "timestamp": ego["timestamp"],
        "translation": ego["translation"],
        "rotation": ego["rotation"],
    }


@app.get("/api/media/sample_data/{sample_data_token}")
def media_sample_data(sample_data_token: str, request: Request):
    nusc: NuScenes = request.app.state.nusc
    try:
        sd = nusc.get("sample_data", sample_data_token)
    except KeyError:
        raise HTTPException(404, "Unknown sample_data") from None
    path = resolve_under_dataroot(Path(nusc.dataroot), sd["filename"])
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime or "application/octet-stream", filename=path.name)


@app.get("/api/clips/frames")
def clip_frames(
    request: Request,
    scene_token: str = Query(...),
    channel: str = Query(..., description="e.g. CAM_FRONT or LIDAR_TOP"),
    max_frames: int = Query(40, ge=1, le=200),
    step: int = Query(2, ge=1, le=20),
):
    """Keyframe chain URLs for slideshow / pseudo-video."""
    nusc: NuScenes = request.app.state.nusc
    try:
        nusc.get("scene", scene_token)
    except KeyError:
        raise HTTPException(404, "Unknown scene") from None
    chain = request.app.state.scene_samples.get(scene_token, [])
    if not chain:
        raise HTTPException(404, "Scene has no samples")
    out = []
    count = 0
    for sample_token in chain[::step]:
        if count >= max_frames:
            break
        s = nusc.get("sample", sample_token)
        if channel not in s["data"]:
            raise HTTPException(400, f"Channel {channel!r} not in sample")
        sd_token = s["data"][channel]
        sd = nusc.get("sample_data", sd_token)
        out.append(
            {
                "sample_token": sample_token,
                "timestamp": s["timestamp"],
                "sample_data_token": sd_token,
                "media_url": f"/api/media/sample_data/{sd_token}",
                "filename": sd["filename"],
            }
        )
        count += 1
    return {"scene_token": scene_token, "channel": channel, "frames": out}


@app.get("/api/search")
def search_samples(
    request: Request,
    category: str | None = Query(None, description="Substring match on category_name"),
    scene_token: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    nusc: NuScenes = request.app.state.nusc
    cache: dict[str, list[str]] = request.app.state.search_cache
    ck = f"{(category or '').lower()}|{scene_token or ''}"
    if ck not in cache:
        ordered: list[str] = []
        seen: set[str] = set()
        for ann in nusc.sample_annotation:
            cat = ann["category_name"]
            if category and category.lower() not in cat.lower():
                continue
            st = ann["sample_token"]
            if st in seen:
                continue
            samp = nusc.get("sample", st)
            if scene_token and samp["scene_token"] != scene_token:
                continue
            seen.add(st)
            ordered.append(st)
        ordered.sort(key=lambda t: nusc.get("sample", t)["timestamp"])
        cache[ck] = ordered
    ordered = cache[ck]
    total = len(ordered)
    start = (page - 1) * page_size
    page_items = ordered[start : start + page_size]
    rows = []
    for st in page_items:
        s = nusc.get("sample", st)
        sc = nusc.get("scene", s["scene_token"])
        rows.append(
            {
                "sample_token": st,
                "timestamp": s["timestamp"],
                "scene_token": s["scene_token"],
                "scene_name": sc["name"],
            }
        )
    return {"total": total, "page": page, "page_size": page_size, "items": rows}


@app.get("/api/render/lidar_bev")
def render_lidar(
    sample_token: str,
    boxes: bool = Query(False, description="Overlay 3D annotation boxes in ego BEV"),
    nusc: NuScenes = Depends(get_nusc),
    cache: RenderLRU = Depends(get_render_cache),
):
    key = f"lidar:{sample_token}:b{int(boxes)}"
    hit = cache.get(key)
    if hit:
        return Response(content=hit, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)
    try:
        png = render_lidar_bev_png(nusc, sample_token, draw_boxes=boxes)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    cache.set(key, png)
    return Response(content=png, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)


@app.get("/api/render/camera_2d")
def render_cam2d(
    sample_token: str,
    channel: str = Query(...),
    boxes: bool = Query(True, description="Draw projected 3D boxes; false returns raw camera"),
    vehicle_only: bool = Query(False, description="If true, only project vehicle categories"),
    nusc: NuScenes = Depends(get_nusc),
    cache: RenderLRU = Depends(get_render_cache),
):
    key = f"cam2d:{sample_token}:{channel}:b{int(boxes)}:v{int(vehicle_only)}:sd"
    hit = cache.get(key)
    if hit:
        return Response(content=hit, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)
    try:
        png = render_camera_with_boxes_png(
            nusc,
            sample_token,
            channel,
            draw_boxes=boxes,
            vehicle_only=vehicle_only,
        )
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    cache.set(key, png)
    return Response(content=png, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)


@app.get("/api/render/ann_bev")
def render_ann_bev(
    sample_token: str,
    nusc: NuScenes = Depends(get_nusc),
    cache: RenderLRU = Depends(get_render_cache),
):
    key = f"annbev:{sample_token}"
    hit = cache.get(key)
    if hit:
        return Response(content=hit, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)
    try:
        png = render_annotation_bev_png(nusc, sample_token)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    cache.set(key, png)
    return Response(content=png, media_type="image/png", headers=_IMAGE_CACHE_HEADERS)


@app.get("/")
def index():
    index_path = _STATIC / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<p>Missing static/index.html</p>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
