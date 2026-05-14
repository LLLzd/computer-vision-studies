"""Resolve dataset files under dataroot with path traversal protection."""

from pathlib import Path

from fastapi import HTTPException


def resolve_under_dataroot(dataroot: Path, relative_filename: str) -> Path:
    """
    Join dataroot with a relative path from NuScenes JSON and ensure the result
    stays inside dataroot (after resolving symlinks).
    """
    if not relative_filename or relative_filename.startswith(".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = dataroot.resolve()
    candidate = (root / relative_filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=403, detail="Path escapes dataroot") from e
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate
