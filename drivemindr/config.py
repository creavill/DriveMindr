"""
DriveMindr configuration — protected paths, settings, thresholds, and logging.

All configuration is local. No network calls, no telemetry, no cloud.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Central structured logger for the entire application.
# Logs are written to a rotating file so the user can share them for debugging.
# Console output uses Rich via typer; the file handler captures everything.

LOG_DIR: Final[Path] = Path(os.environ.get("DRIVEMINDR_LOG_DIR", "."))
LOG_FILE: Final[str] = "drivemindr_debug.log"
LOG_MAX_BYTES: Final[int] = 5 * 1024 * 1024  # 5 MB per file
LOG_BACKUP_COUNT: Final[int] = 3  # Keep 3 rotated files


def setup_logging(*, verbose: bool = False, log_dir: Path | None = None) -> None:
    """Configure application-wide logging.

    - File handler: always DEBUG level, rotating, structured for sharing.
    - Console handler: INFO by default, DEBUG with ``--verbose``.
    """
    resolved_dir = log_dir or LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / LOG_FILE

    root = logging.getLogger("drivemindr")
    root.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if root.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler — always captures full DEBUG output
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — respects verbosity flag
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


logger = logging.getLogger("drivemindr")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DEFAULT_DB_NAME: Final[str] = "drivemindr.db"

# ---------------------------------------------------------------------------
# Ollama (localhost only — NEVER changes)
# ---------------------------------------------------------------------------
OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"
OLLAMA_MODEL: Final[str] = "llama3.1:8b"
OLLAMA_BATCH_SIZE: Final[int] = 50  # files per classification request

# ---------------------------------------------------------------------------
# AI confidence thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_AUTO_APPROVE: Final[float] = 0.7
CONFIDENCE_UNCERTAIN: Final[float] = 0.4
CONFIDENCE_DELETE_MIN: Final[float] = 0.85

# ---------------------------------------------------------------------------
# Protected paths — HARDCODED, AI CANNOT OVERRIDE
# ---------------------------------------------------------------------------
# These use lower-case normalized representations.  Every path checked against
# this list is compared via ``PureWindowsPath`` case-insensitive matching.
PROTECTED_PATHS: Final[list[str]] = [
    r"C:\Windows",
    r"C:\Program Files\WindowsApps",
    r"C:\Program Files\Windows Defender",
    r"C:\Program Files\Windows Defender Advanced Threat Protection",
    r"C:\Program Files\Windows Mail",
    r"C:\Program Files\Windows Media Player",
    r"C:\Program Files\Windows Multimedia Platform",
    r"C:\Program Files\Windows NT",
    r"C:\Program Files\Windows Photo Viewer",
    r"C:\Program Files\Windows Portable Devices",
    r"C:\Program Files\Windows Security",
    r"C:\Program Files\Windows Sidebar",
    r"C:\ProgramData\Microsoft",
    r"C:\Program Files (x86)\Windows Defender",
    r"C:\Recovery",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
    r"C:\Boot",
    r"C:\bootmgr",
    r"C:\BOOTNXT",
]

# Ownership that makes a file untouchable regardless of path
PROTECTED_OWNERS: Final[list[str]] = [
    "TrustedInstaller",
    "NT SERVICE\\TrustedInstaller",
    "SYSTEM",
    "NT AUTHORITY\\SYSTEM",
]

# ---------------------------------------------------------------------------
# Document Guardian — file extensions that may NEVER be auto-deleted
# ---------------------------------------------------------------------------
DOCUMENT_EXTENSIONS: Final[frozenset[str]] = frozenset({
    # Text documents
    ".doc", ".docx", ".pdf", ".txt", ".md", ".rtf", ".odt", ".tex", ".pages",
    # Spreadsheets
    ".xls", ".xlsx", ".csv", ".ods", ".numbers",
    # Presentations
    ".ppt", ".pptx", ".odp", ".key",
    # eBooks
    ".epub", ".mobi",
})

PHOTO_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({
    # Photos
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".svg", ".raw", ".cr2", ".nef", ".heic", ".heif",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    # Audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a",
})

SOURCE_CODE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".r", ".m", ".sql", ".sh", ".bash", ".ps1", ".bat",
    ".cmd", ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".less", ".vue", ".svelte",
})

SENSITIVE_FILE_PATTERNS: Final[list[str]] = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "_key",
    "_secret",
    "_token",
    "credentials",
    "secret",
    "private_key",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
    ".pfx",
    ".p12",
]

# All extensions that the Document Guardian protects from deletion
GUARDIAN_EXTENSIONS: Final[frozenset[str]] = (
    DOCUMENT_EXTENSIONS | PHOTO_VIDEO_EXTENSIONS | SOURCE_CODE_EXTENSIONS
)

# ---------------------------------------------------------------------------
# D: drive target organization structure
# ---------------------------------------------------------------------------
D_DRIVE_STRUCTURE: Final[dict[str, str]] = {
    "apps": r"D:\Apps",
    "documents": r"D:\Documents",
    "media_photos": r"D:\Media\Photos",
    "media_videos": r"D:\Media\Videos",
    "media_music": r"D:\Media\Music",
    "projects": r"D:\Projects",
    "archive": r"D:\Archive",
    "drivemindr": r"D:\DriveMindr",
}

# ---------------------------------------------------------------------------
# Scanner defaults
# ---------------------------------------------------------------------------
SCANNER_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    "$Recycle.Bin",
    "System Volume Information",
    "$WinREAgent",
    "$SysReset",
})
