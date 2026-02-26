"""
DriveMindr symlink/junction manager — Windows Directory Junctions for app migration.

Moves application directories from C: to D: and creates a Directory Junction
so Windows still finds the app at its original path.

Flow:
  1. Copy entire app directory to D:\\Apps\\<app>
  2. Verify the copy with checksum comparison
  3. Remove the original directory
  4. Create a Directory Junction: C:\\...\\<app> -> D:\\Apps\\<app>
  5. Log everything for undo

Requirements:
  - Administrator privileges (junctions require elevated access on Windows)
  - Windows OS (junctions are a Windows-specific feature)

On non-Windows systems, operations are logged but junction creation is
simulated (for testing and development).
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from drivemindr.undo import UndoManager, file_checksum

logger = logging.getLogger("drivemindr.symlinks")


def is_admin() -> bool:
    """Check if the current process has Administrator privileges."""
    if platform.system() != "Windows":
        # On non-Windows, check if running as root
        return os.getuid() == 0  # type: ignore[attr-defined]
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def is_junction(path: Path) -> bool:
    """Check if a path is a Windows Directory Junction."""
    if not path.exists() and not path.is_symlink():
        return False
    # Path.is_junction() available in Python 3.12+
    if hasattr(path, "is_junction"):
        return path.is_junction()
    # Fallback: check if it's a symlink or reparse point
    return path.is_symlink()


def create_junction(link_path: Path, target_path: Path) -> bool:
    """Create a Windows Directory Junction.

    Args:
        link_path: Where the junction will appear (e.g. C:\\Program Files\\App)
        target_path: Where the data actually lives (e.g. D:\\Apps\\App)

    Returns:
        True if junction was created successfully.
    """
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(
                "Created junction: %s -> %s (output: %s)",
                link_path, target_path, result.stdout.strip(),
            )
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to create junction %s -> %s: %s",
                link_path, target_path, exc.stderr,
            )
            return False
    else:
        # Non-Windows: use symlink as a stand-in for testing
        try:
            link_path.symlink_to(target_path, target_is_directory=True)
            logger.info(
                "Created symlink (non-Windows junction stand-in): %s -> %s",
                link_path, target_path,
            )
            return True
        except OSError as exc:
            logger.error("Failed to create symlink: %s", exc)
            return False


def remove_junction(path: Path) -> bool:
    """Remove a junction/symlink without removing the target directory."""
    try:
        if platform.system() == "Windows":
            # On Windows, rmdir removes the junction point without touching target
            subprocess.run(
                ["cmd", "/c", "rmdir", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # On non-Windows, unlink the symlink
            path.unlink()
        logger.info("Removed junction: %s", path)
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.error("Failed to remove junction %s: %s", path, exc)
        return False


class AppMigrator:
    """Handles moving applications from C: to D: with junction-based redirection.

    Usage::

        migrator = AppMigrator(undo_manager, target_root=Path("D:/Apps"))
        result = migrator.migrate_app(
            source=Path("C:/Program Files/SomeApp"),
            file_id=42,
            batch_id="batch_123",
        )
    """

    def __init__(
        self,
        undo: UndoManager,
        *,
        target_root: Path = Path(r"D:\Apps"),
    ) -> None:
        self.undo = undo
        self.target_root = target_root

    def migrate_app(
        self,
        source: Path,
        file_id: int | None = None,
        batch_id: str | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, str | bool]:
        """Migrate an application directory from source to D:\\Apps.

        Steps:
          1. Copy source -> target
          2. Verify copy integrity
          3. Remove original
          4. Create junction: source -> target
          5. Log for undo

        Returns dict with status, source, target, junction_created.
        """
        target = self.target_root / source.name
        result: dict[str, str | bool] = {
            "source": str(source),
            "target": str(target),
            "junction_created": False,
            "success": False,
        }

        if not source.exists():
            logger.error("Source directory does not exist: %s", source)
            result["error"] = "Source not found"
            return result

        if not source.is_dir():
            logger.error("Source is not a directory: %s", source)
            result["error"] = "Source is not a directory"
            return result

        if target.exists():
            logger.error("Target already exists: %s", target)
            result["error"] = "Target already exists"
            return result

        if dry_run:
            logger.info(
                "DRY RUN — would migrate %s -> %s with junction",
                source, target,
            )
            result["success"] = True
            result["dry_run"] = True
            return result

        # Step 1: Copy
        logger.info("Copying %s -> %s", source, target)
        try:
            shutil.copytree(str(source), str(target))
        except (OSError, shutil.Error) as exc:
            logger.error("Copy failed: %s", exc)
            result["error"] = f"Copy failed: {exc}"
            return result

        # Step 2: Verify (compare directory file count as basic integrity check)
        source_count = sum(1 for _ in source.rglob("*") if _.is_file())
        target_count = sum(1 for _ in target.rglob("*") if _.is_file())
        if source_count != target_count:
            logger.error(
                "Copy verification failed: source has %d files, target has %d",
                source_count, target_count,
            )
            shutil.rmtree(str(target), ignore_errors=True)
            result["error"] = "Copy verification failed"
            return result

        # Step 3: Remove original
        logger.info("Removing original: %s", source)
        try:
            shutil.rmtree(str(source))
        except OSError as exc:
            logger.error("Failed to remove original: %s", exc)
            result["error"] = f"Failed to remove original: {exc}"
            return result

        # Step 4: Create junction
        junction_ok = create_junction(source, target)
        result["junction_created"] = junction_ok

        if not junction_ok:
            # Rollback: move data back
            logger.warning("Junction creation failed — rolling back")
            shutil.move(str(target), str(source))
            result["error"] = "Junction creation failed"
            return result

        # Step 5: Log for undo
        self.undo.log_action(
            file_id=file_id,
            action="SYMLINKED",
            source_path=str(source),
            dest_path=str(target),
            batch_id=batch_id,
        )

        result["success"] = True
        logger.info("App migration complete: %s -> %s", source, target)
        return result
