"""
DriveMindr execution engine — carries out user-approved file operations.

Reads approved actions from the database and executes them:
  - MOVE_DATA / MOVE_APP: move files to organized D: folders
  - DELETE_JUNK / DELETE_UNUSED: move to trash (not permanent delete)
  - ARCHIVE: compress files into zip archives

Every operation is:
  1. Checksummed before execution
  2. Logged to the undo system
  3. Verified after execution (for moves)
  4. Skippable via --dry-run

Nothing is permanently deleted — "delete" operations move files to a
DriveMindr trash directory so they can be restored via undo.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from drivemindr.config import (
    D_DRIVE_STRUCTURE,
    DOCUMENT_EXTENSIONS,
    PHOTO_VIDEO_EXTENSIONS,
    SOURCE_CODE_EXTENSIONS,
)
from drivemindr.database import Database
from drivemindr.symlinks import AppMigrator
from drivemindr.undo import UndoManager, file_checksum

logger = logging.getLogger("drivemindr.executor")


def _categorize_destination(path: str, extension: str) -> str:
    """Determine the D: drive destination category for a file.

    Maps file metadata to the appropriate D_DRIVE_STRUCTURE key.
    """
    ext = extension.lower() if extension else ""
    path_lower = path.lower()

    if ext in DOCUMENT_EXTENSIONS:
        return "documents"
    if ext in PHOTO_VIDEO_EXTENSIONS:
        # Subcategorize photos/videos/music
        if ext in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"}:
            return "media_music"
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}:
            return "media_videos"
        return "media_photos"
    if ext in SOURCE_CODE_EXTENSIONS:
        return "projects"
    # Check path hints
    if "project" in path_lower or "repos" in path_lower or "github" in path_lower:
        return "projects"
    return "documents"  # safe default


def _compute_dest_path(source_path: str, category: str) -> Path:
    """Compute the destination path on D:, preserving subfolder structure.

    Strips the drive letter and top-level folder, then appends to the
    D: category folder. E.g.:
      C:\\Users\\Conner\\Documents\\Work\\report.pdf
      -> D:\\Documents\\Work\\report.pdf

    Uses PureWindowsPath to correctly parse Windows paths on any OS.
    """
    from pathlib import PureWindowsPath

    src = PureWindowsPath(source_path)
    base = D_DRIVE_STRUCTURE.get(category, D_DRIVE_STRUCTURE["documents"])

    # Try to preserve meaningful subfolder structure
    # Strip drive root and first 2 path components (e.g. C:\, Users, Username)
    parts = src.parts
    if len(parts) > 3:
        # Skip drive + top-level dirs
        relative = Path(*parts[3:])
    else:
        relative = Path(src.name)

    return Path(base) / relative


class ExecutionEngine:
    """Orchestrates file operations from user-approved action plan.

    Usage::

        engine = ExecutionEngine(db)
        summary = engine.execute_plan(dry_run=True)  # preview first
        summary = engine.execute_plan()               # then execute
    """

    def __init__(
        self,
        db: Database,
        *,
        undo: UndoManager | None = None,
        app_migrator: AppMigrator | None = None,
        trash_dir: Path | None = None,
    ) -> None:
        self.db = db
        self._trash_dir = trash_dir or Path(r"D:\DriveMindr\trash")
        self.undo = undo or UndoManager(db, trash_dir=self._trash_dir)
        self.app_migrator = app_migrator or AppMigrator(self.undo)

        self._moved = 0
        self._deleted = 0
        self._archived = 0
        self._symlinked = 0
        self._skipped = 0
        self._errors = 0

    def execute_plan(
        self,
        *,
        dry_run: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Execute all user-approved actions.

        Args:
            dry_run: If True, log what would happen but change nothing.
            progress_callback: Optional callable(moved, deleted, archived, errors).

        Returns:
            Summary dict with operation counts and batch_id.
        """
        approved = self.db.get_approved_actions()
        if not approved:
            logger.info("No approved actions to execute.")
            return self._summary(batch_id=None)

        batch_id = self.undo.new_batch()
        logger.info(
            "Executing %d approved actions — batch=%s%s",
            len(approved), batch_id, " (DRY RUN)" if dry_run else "",
        )

        for row in approved:
            file_id = row["id"]
            action = row["final_action"]
            source_path = row["path"]
            extension = row["extension"]

            try:
                if action in ("MOVE_DATA", "MOVE_APP"):
                    self._execute_move(
                        file_id, source_path, extension, action,
                        batch_id, dry_run=dry_run,
                    )
                elif action in ("DELETE_JUNK", "DELETE_UNUSED"):
                    self._execute_delete(
                        file_id, source_path, batch_id, dry_run=dry_run,
                    )
                elif action == "ARCHIVE":
                    self._execute_archive(
                        file_id, source_path, batch_id, dry_run=dry_run,
                    )
                else:
                    logger.debug("Skipping action %s for %s", action, source_path)
                    self._skipped += 1
            except Exception:
                logger.exception("Error executing %s on %s", action, source_path)
                self._errors += 1

            if progress_callback:
                progress_callback(
                    self._moved, self._deleted, self._archived, self._errors,
                )

        summary = self._summary(batch_id=batch_id)
        logger.info("Execution complete: %s", summary)
        return summary

    def _summary(self, batch_id: str | None) -> dict[str, Any]:
        return {
            "batch_id": batch_id,
            "moved": self._moved,
            "deleted": self._deleted,
            "archived": self._archived,
            "symlinked": self._symlinked,
            "skipped": self._skipped,
            "errors": self._errors,
        }

    # -- individual operations -------------------------------------------------

    def _execute_move(
        self,
        file_id: int,
        source_path: str,
        extension: str,
        action: str,
        batch_id: str,
        *,
        dry_run: bool,
    ) -> None:
        """Move a file to the organized D: structure."""
        src = Path(source_path)

        if action == "MOVE_APP" and src.is_dir():
            # App migration uses symlinks
            result = self.app_migrator.migrate_app(
                src, file_id=file_id, batch_id=batch_id, dry_run=dry_run,
            )
            if result.get("success"):
                self._symlinked += 1
            else:
                self._errors += 1
            return

        category = _categorize_destination(source_path, extension)
        dest = _compute_dest_path(source_path, category)

        if dry_run:
            logger.info("DRY RUN — move: %s -> %s", src, dest)
            self._moved += 1
            return

        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            self._skipped += 1
            return

        # Checksum before
        checksum_before = file_checksum(src) if src.is_file() else None

        # Move
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

        # Checksum after (verify integrity)
        checksum_after = file_checksum(dest) if dest.is_file() else None
        if checksum_before and checksum_after and checksum_before != checksum_after:
            logger.error(
                "Checksum mismatch after move! %s: %s != %s",
                source_path, checksum_before, checksum_after,
            )
            # Rollback this one move
            shutil.move(str(dest), str(src))
            self._errors += 1
            return

        # Log for undo
        self.undo.log_action(
            file_id=file_id,
            action="MOVED",
            source_path=source_path,
            dest_path=str(dest),
            batch_id=batch_id,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
        )
        self._moved += 1
        logger.info("Moved: %s -> %s", src, dest)

    def _execute_delete(
        self,
        file_id: int,
        source_path: str,
        batch_id: str,
        *,
        dry_run: bool,
    ) -> None:
        """'Delete' a file by moving it to the DriveMindr trash.

        Files are NEVER permanently deleted — they go to trash so undo works.
        """
        src = Path(source_path)
        trash_dest = self.undo.get_trash_path(src, batch_id)

        if dry_run:
            logger.info("DRY RUN — delete (to trash): %s -> %s", src, trash_dest)
            self._deleted += 1
            return

        if not src.exists():
            logger.warning("Source not found, skipping delete: %s", src)
            self._skipped += 1
            return

        # Checksum before
        checksum_before = file_checksum(src) if src.is_file() else None

        # Move to trash
        trash_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(trash_dest))

        # Log for undo
        self.undo.log_action(
            file_id=file_id,
            action="DELETED",
            source_path=source_path,
            dest_path=str(trash_dest),
            batch_id=batch_id,
            checksum_before=checksum_before,
        )
        self._deleted += 1
        logger.info("Deleted (to trash): %s -> %s", src, trash_dest)

    def _execute_archive(
        self,
        file_id: int,
        source_path: str,
        batch_id: str,
        *,
        dry_run: bool,
    ) -> None:
        """Archive a file into a zip in D:\\Archive\\YYYY-MM\\."""
        src = Path(source_path)
        now = datetime.now()
        archive_dir = Path(D_DRIVE_STRUCTURE["archive"]) / now.strftime("%Y-%m")
        archive_name = f"{src.stem}.zip"
        archive_path = archive_dir / archive_name

        if dry_run:
            logger.info("DRY RUN — archive: %s -> %s", src, archive_path)
            self._archived += 1
            return

        if not src.exists():
            logger.warning("Source not found, skipping archive: %s", src)
            self._skipped += 1
            return

        # Checksum before
        checksum_before = file_checksum(src) if src.is_file() else None

        # Create archive
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Handle name collisions
        counter = 1
        while archive_path.exists():
            archive_path = archive_dir / f"{src.stem}_{counter}.zip"
            counter += 1

        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                if src.is_file():
                    zf.write(str(src), src.name)
                elif src.is_dir():
                    for fp in src.rglob("*"):
                        if fp.is_file():
                            zf.write(str(fp), str(fp.relative_to(src.parent)))
        except (OSError, zipfile.BadZipFile) as exc:
            logger.error("Archive creation failed for %s: %s", src, exc)
            self._errors += 1
            return

        # Log for undo (keep originals — archive is additive)
        self.undo.log_action(
            file_id=file_id,
            action="ARCHIVED",
            source_path=source_path,
            dest_path=str(archive_path),
            batch_id=batch_id,
            checksum_before=checksum_before,
        )
        self._archived += 1
        logger.info("Archived: %s -> %s", src, archive_path)
