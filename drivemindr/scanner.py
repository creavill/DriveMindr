"""
DriveMindr file scanner — recursive metadata collection and Windows registry app detection.

Walks the target drive(s), collects metadata, and stores everything in SQLite.
Never reads file contents. Never makes network calls.
"""

from __future__ import annotations

import datetime
import logging
import os
import platform
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any

from drivemindr.config import PROTECTED_PATHS, SCANNER_SKIP_DIRS
from drivemindr.database import Database

logger = logging.getLogger("drivemindr.scanner")


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _timestamp(epoch: float | None) -> str | None:
    """Convert an epoch float to an ISO-format local timestamp string."""
    if epoch is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(epoch).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _get_file_owner(path: Path) -> str | None:
    """Get the file owner on Windows. Returns None on other platforms or on error."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]

        # GetFileSecurity → SecurityDescriptor → LookupAccountSid
        SE_FILE_OBJECT = 1
        OWNER_SECURITY_INFORMATION = 0x00000001

        path_str = str(path)
        size_needed = wintypes.DWORD(0)

        advapi32.GetFileSecurityW(
            path_str, OWNER_SECURITY_INFORMATION, None, 0, ctypes.byref(size_needed)
        )
        buf = ctypes.create_string_buffer(size_needed.value)
        if not advapi32.GetFileSecurityW(
            path_str,
            OWNER_SECURITY_INFORMATION,
            buf,
            size_needed.value,
            ctypes.byref(size_needed),
        ):
            return None

        sid = ctypes.c_void_p()
        defaulted = wintypes.BOOL()
        if not advapi32.GetSecurityDescriptorOwner(
            buf, ctypes.byref(sid), ctypes.byref(defaulted)
        ):
            return None

        name = ctypes.create_unicode_buffer(256)
        name_size = wintypes.DWORD(256)
        domain = ctypes.create_unicode_buffer(256)
        domain_size = wintypes.DWORD(256)
        sid_type = wintypes.DWORD()

        if advapi32.LookupAccountSidW(
            None,
            sid,
            name,
            ctypes.byref(name_size),
            domain,
            ctypes.byref(domain_size),
            ctypes.byref(sid_type),
        ):
            owner = f"{domain.value}\\{name.value}" if domain.value else name.value
            return owner
    except Exception:
        logger.debug("Could not get owner for %s", path, exc_info=True)
    return None


def _should_skip_dir(dir_name: str) -> bool:
    """Check if a directory name should be skipped during scanning."""
    return dir_name in SCANNER_SKIP_DIRS


def _is_under_protected_path(file_path: str) -> bool:
    """Check if a path is under any hardcoded protected path."""
    # Normalize to Windows path representation for comparison
    normalized = PureWindowsPath(file_path)
    for protected in PROTECTED_PATHS:
        protected_p = PureWindowsPath(protected)
        try:
            normalized.relative_to(protected_p)
            return True
        except ValueError:
            continue
    return False


def _collect_metadata(entry: os.DirEntry[str], scan_id: str) -> dict[str, Any] | None:
    """Collect metadata for a single file/directory entry.

    Returns None if metadata cannot be read (permission denied, etc.).
    """
    try:
        stat = entry.stat(follow_symlinks=False)
        path = Path(entry.path)
        return {
            "path": str(path),
            "name": entry.name,
            "extension": path.suffix.lower() if not entry.is_dir(follow_symlinks=False) else None,
            "size_bytes": stat.st_size if not entry.is_dir(follow_symlinks=False) else 0,
            "created": _timestamp(stat.st_ctime),
            "modified": _timestamp(stat.st_mtime),
            "accessed": _timestamp(stat.st_atime),
            "owner": _get_file_owner(path),
            "is_readonly": 1 if not os.access(entry.path, os.W_OK) else 0,
            "is_dir": 1 if entry.is_dir(follow_symlinks=False) else 0,
            "parent_dir": str(path.parent),
            "scan_id": scan_id,
        }
    except PermissionError:
        logger.warning("Permission denied: %s", entry.path)
        return None
    except OSError as exc:
        logger.warning("OS error scanning %s: %s", entry.path, exc)
        return None


class FileScanner:
    """Recursively scans a drive and populates the database with file metadata."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.scan_id = str(uuid.uuid4())[:8]
        self._file_count = 0
        self._dir_count = 0
        self._error_count = 0
        self._total_bytes = 0
        # Track per-directory sizes for aggregation
        self._dir_sizes: dict[str, tuple[int, int]] = {}  # path → (bytes, count)

    # -- public API -----------------------------------------------------------

    def scan(self, root: str | Path, *, progress_callback: Any = None) -> dict[str, int]:
        """Scan *root* recursively and store metadata in the database.

        Args:
            root: The top-level path to scan (e.g. ``C:\\``).
            progress_callback: Optional callable(scanned: int, errors: int)
                invoked periodically so the caller can show progress.

        Returns:
            Summary dict with ``files``, ``dirs``, ``errors``, ``total_bytes``.
        """
        root_path = Path(root)
        logger.info(
            "Starting scan — root=%s scan_id=%s", root_path, self.scan_id
        )

        if not root_path.exists():
            logger.error("Scan root does not exist: %s", root_path)
            raise FileNotFoundError(f"Scan root does not exist: {root_path}")

        self._walk(str(root_path), progress_callback)

        # Flush dir size aggregations
        for dir_path, (total_bytes, file_count) in self._dir_sizes.items():
            self.db.upsert_dir_size(dir_path, total_bytes, file_count, self.scan_id)

        summary = {
            "files": self._file_count,
            "dirs": self._dir_count,
            "errors": self._error_count,
            "total_bytes": self._total_bytes,
            "scan_id": self.scan_id,
        }
        logger.info(
            "Scan complete — files=%d dirs=%d errors=%d bytes=%d",
            self._file_count,
            self._dir_count,
            self._error_count,
            self._total_bytes,
        )
        return summary

    def scan_installed_apps(self) -> int:
        """Scan Windows Registry for installed applications.

        Returns the number of apps found. No-op on non-Windows.
        """
        if not _is_windows():
            logger.info("Skipping registry scan — not running on Windows")
            return 0

        count = 0
        try:
            import winreg

            uninstall_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]

            for hive, key_path in uninstall_keys:
                try:
                    key = winreg.OpenKey(hive, key_path)
                except OSError:
                    logger.debug("Registry key not found: %s", key_path)
                    continue

                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        subkey_path = f"{key_path}\\{subkey_name}"
                        subkey = winreg.OpenKey(hive, subkey_path)

                        def _read(name: str) -> str | None:
                            try:
                                val, _ = winreg.QueryValueEx(subkey, name)
                                return str(val) if val else None
                            except OSError:
                                return None

                        display_name = _read("DisplayName")
                        if display_name:
                            app_data = {
                                "display_name": display_name,
                                "install_location": _read("InstallLocation"),
                                "publisher": _read("Publisher"),
                                "install_date": _read("InstallDate"),
                                "estimated_size": None,
                                "uninstall_string": _read("UninstallString"),
                                "registry_key": subkey_path,
                            }
                            size_str = _read("EstimatedSize")
                            if size_str and size_str.isdigit():
                                app_data["estimated_size"] = int(size_str) * 1024  # KB → bytes

                            self.db.upsert_installed_app(app_data)
                            count += 1

                        winreg.CloseKey(subkey)
                        i += 1
                    except OSError:
                        break
                winreg.CloseKey(key)

        except ImportError:
            logger.info("winreg not available — skipping registry scan")
        except Exception:
            logger.exception("Error during registry scan")

        logger.info("Registry scan complete — found %d installed apps", count)
        return count

    # -- internal -------------------------------------------------------------

    def _walk(self, root: str, progress_callback: Any) -> None:
        """Walk the directory tree using os.scandir for performance."""
        batch: list[dict[str, Any]] = []
        batch_size = 500

        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if _should_skip_dir(entry.name):
                                logger.debug("Skipping dir: %s", entry.path)
                                continue
                            self._dir_count += 1
                            # Record the directory itself
                            meta = _collect_metadata(entry, self.scan_id)
                            if meta:
                                batch.append(meta)
                            # Recurse
                            self._walk(entry.path, progress_callback)
                        else:
                            meta = _collect_metadata(entry, self.scan_id)
                            if meta:
                                self._file_count += 1
                                self._total_bytes += meta["size_bytes"]
                                batch.append(meta)

                                # Track per-parent-dir sizes
                                parent = meta["parent_dir"]
                                cur_bytes, cur_count = self._dir_sizes.get(parent, (0, 0))
                                self._dir_sizes[parent] = (
                                    cur_bytes + meta["size_bytes"],
                                    cur_count + 1,
                                )

                    except PermissionError:
                        self._error_count += 1
                        logger.warning("Permission denied during walk: %s", entry.path)
                    except OSError as exc:
                        self._error_count += 1
                        logger.warning("OS error during walk: %s — %s", entry.path, exc)

                    # Flush batch
                    if len(batch) >= batch_size:
                        self.db.bulk_upsert_files(batch)
                        batch.clear()
                        if progress_callback:
                            progress_callback(self._file_count, self._error_count)

        except PermissionError:
            self._error_count += 1
            logger.warning("Permission denied opening dir: %s", root)
        except OSError as exc:
            self._error_count += 1
            logger.warning("OS error opening dir: %s — %s", root, exc)

        # Flush remaining
        if batch:
            self.db.bulk_upsert_files(batch)
            if progress_callback:
                progress_callback(self._file_count, self._error_count)
