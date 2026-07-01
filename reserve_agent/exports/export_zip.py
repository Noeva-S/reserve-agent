from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


def _safe_zip_name(filename: str) -> str:
    path = PurePosixPath(str(filename).replace("\\", "/"))
    parts = [part for part in path.parts if part not in {"", ".", ".."}]
    safe = PurePosixPath(*parts).as_posix() if parts else "unnamed.bin"
    return safe.lstrip("/")


def build_zip_package(files: dict[str, bytes], *, processing_log: str | None = None) -> bytes:
    """Build a ZIP package from filename-to-bytes mapping.

    The function sanitises paths and skips empty payloads, so UI code can pass
    optional chart/report files without risking a broken archive.
    """

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        written: set[str] = set()
        for filename, content in files.items():
            if content is None:
                continue
            safe_name = _safe_zip_name(filename)
            if safe_name in written:
                stem, dot, suffix = safe_name.partition(".")
                safe_name = f"{stem}_{len(written)}{dot}{suffix}" if dot else f"{safe_name}_{len(written)}"
            zf.writestr(safe_name, content)
            written.add(safe_name)
        if processing_log:
            zf.writestr("processing_log.txt", processing_log)
    return buffer.getvalue()
