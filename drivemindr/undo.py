"""
DriveMindr undo system — action logging and rollback.

Every file operation (move, delete, archive, symlink) is logged before
execution. Batches of operations share a batch_id so they can be undone
together. The undo system can reverse moves (move back), restore deleted
files (from a safety backup), remove archives, and remove junctions.

Design:
  - Deleted files are NOT permanently removed — they're moved to a
    DriveMindr trash folder first, so undo can restore them.
  - Every action is logged with source/dest paths and checksums.
  - Undo marks log entries as undone=1 rather than deleting them.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from drivemindr.database import Database

logger = logging.getLogger("drivemindr.undo")

# Trash location for "deleted" files (so they can be restored)
DEFAULT_TRASH_DIR = Path(r"D:\DriveMindr\trash")


def generate_batch_id() -> str:
    """Generate a unique batch ID for grouping related operations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"batch_{timestamp}_{short_uuid}"


def file_checksum(path: Path, algorithm: str = "sha256") -> str | None:
    """Compute a file checksum. Returns None if file doesn't exist or is unreadable."""
    try:
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError) as exc:
        logger.warning("Could not checksum %s: %s", path, exc)
        return None


class UndoManager:
    """Manages action logging and rollback operations.

    Usage::

        undo = UndoManager(db, trash_dir=Path("D:/DriveMindr/trash"))
        batch_id = undo.new_batch()

        # Log before executing
        undo.log_action(file_id, "MOVED", source, dest, batch_id)

        # Undo a whole batch
        undo.undo_batch(batch_id)
    """

    def __init__(
        self,
        db: Database,
        *,
        trash_dir: Path = DEFAULT_TRASH_DIR,
    ) -> None:
        self.db = db
        self.trash_dir = trash_dir

    def new_batch(self) -> str:
        """Create a new batch ID."""
        batch_id = generate_batch_id()
        logger.info("New undo batch: %s", batch_id)
        return batch_id

    def log_action(
        self,
        file_id: int | None,
        action: str,
        source_path: str,
        dest_path: str | None = None,
        batch_id: str | None = None,
        checksum_before: str | None = None,
        checksum_after: str | None = None,
    ) -> int:
        """Log an action to the action_log table. Returns the log entry ID."""
        sql = """
        INSERT INTO action_log
            (file_id, action, source_path, dest_path, checksum_before,
             checksum_after, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (
                file_id, action, source_path, dest_path,
                checksum_before, checksum_after, batch_id,
            ))
            log_id = cur.lastrowid
        logger.debug(
            "Logged action #%d: %s %s -> %s (batch=%s)",
            log_id, action, source_path, dest_path, batch_id,
        )
        return log_id  # type: ignore[return-value]

    def get_batch_actions(self, batch_id: str) -> list[dict[str, Any]]:
        """Get all actions in a batch, ordered for undo (reverse execution order)."""
        sql = """
        SELECT * FROM action_log
        WHERE batch_id = ? AND undone = 0
        ORDER BY id DESC
        """
        rows = self.db.conn.execute(sql, (batch_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_recent_batches(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent batch IDs with action counts."""
        sql = """
        SELECT batch_id, COUNT(*) AS action_count,
               MIN(executed_at) AS started_at,
               MAX(executed_at) AS finished_at,
               SUM(undone) AS undone_count
        FROM action_log
        WHERE batch_id IS NOT NULL
        GROUP BY batch_id
        ORDER BY MAX(id) DESC
        LIMIT ?
        """
        rows = self.db.conn.execute(sql, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def undo_batch(self, batch_id: str, *, dry_run: bool = False) -> dict[str, int]:
        """Undo all actions in a batch (in reverse order).

        Returns summary with counts of undone, skipped, and failed actions.
        """
        actions = self.get_batch_actions(batch_id)
        if not actions:
            logger.warning("No undoable actions found for batch %s", batch_id)
            return {"undone": 0, "skipped": 0, "failed": 0}

        logger.info(
            "Undoing batch %s — %d actions%s",
            batch_id, len(actions), " (DRY RUN)" if dry_run else "",
        )

        undone = 0
        skipped = 0
        failed = 0

        for entry in actions:
            try:
                success = self._undo_single(entry, dry_run=dry_run)
                if success:
                    undone += 1
                else:
                    skipped += 1
            except Exception:
                logger.exception("Failed to undo action #%d", entry["id"])
                failed += 1

        summary = {"undone": undone, "skipped": skipped, "failed": failed}
        logger.info("Undo batch %s complete: %s", batch_id, summary)
        return summary

    def _undo_single(self, entry: dict[str, Any], *, dry_run: bool = False) -> bool:
        """Undo a single logged action. Returns True if undone."""
        action = entry["action"]
        source = Path(entry["source_path"])
        dest = Path(entry["dest_path"]) if entry["dest_path"] else None
        log_id = entry["id"]

        logger.debug("Undoing action #%d: %s", log_id, action)

        if action == "MOVED":
            return self._undo_move(log_id, source, dest, dry_run=dry_run)
        elif action == "DELETED":
            return self._undo_delete(log_id, source, dest, dry_run=dry_run)
        elif action == "ARCHIVED":
            return self._undo_archive(log_id, source, dest, dry_run=dry_run)
        elif action == "SYMLINKED":
            return self._undo_symlink(log_id, source, dest, dry_run=dry_run)
        else:
            logger.warning("Unknown action type '%s' — skipping", action)
            return False

    def _undo_move(
        self, log_id: int, source: Path, dest: Path | None, *, dry_run: bool,
    ) -> bool:
        """Undo a move: move file back from dest to source."""
        if dest is None or not dest.exists():
            logger.warning("Cannot undo move #%d — dest %s not found", log_id, dest)
            return False

        if not dry_run:
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(source))
            self._mark_undone(log_id)
        logger.info("Undo move: %s -> %s%s", dest, source, " (dry)" if dry_run else "")
        return True

    def _undo_delete(
        self, log_id: int, source: Path, dest: Path | None, *, dry_run: bool,
    ) -> bool:
        """Undo a delete: restore from trash (dest) back to source."""
        if dest is None or not dest.exists():
            logger.warning("Cannot undo delete #%d — trash copy %s not found", log_id, dest)
            return False

        if not dry_run:
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(source))
            self._mark_undone(log_id)
        logger.info("Undo delete: restored %s%s", source, " (dry)" if dry_run else "")
        return True

    def _undo_archive(
        self, log_id: int, source: Path, dest: Path | None, *, dry_run: bool,
    ) -> bool:
        """Undo an archive: remove the archive file (originals are kept)."""
        if dest is None or not dest.exists():
            logger.warning("Cannot undo archive #%d — archive %s not found", log_id, dest)
            return False

        if not dry_run:
            dest.unlink()
            self._mark_undone(log_id)
        logger.info("Undo archive: removed %s%s", dest, " (dry)" if dry_run else "")
        return True

    def _undo_symlink(
        self, log_id: int, source: Path, dest: Path | None, *, dry_run: bool,
    ) -> bool:
        """Undo a symlink/junction: remove junction, move data back."""
        if dest is None:
            logger.warning("Cannot undo symlink #%d — no dest recorded", log_id)
            return False

        if not dry_run:
            # Remove the junction at source (if it exists and is a junction)
            if source.exists():
                if source.is_symlink() or source.is_junction():
                    source.unlink()
                elif source.is_dir():
                    # Junction might show as dir on some systems
                    source.rmdir()

            # Move data back from dest to source
            if dest.exists():
                shutil.move(str(dest), str(source))

            self._mark_undone(log_id)

        logger.info(
            "Undo symlink: removed junction %s, restored from %s%s",
            source, dest, " (dry)" if dry_run else "",
        )
        return True

    def _mark_undone(self, log_id: int) -> None:
        """Mark an action log entry as undone."""
        sql = "UPDATE action_log SET undone = 1 WHERE id = ?"
        with self.db.transaction() as cur:
            cur.execute(sql, (log_id,))

    def get_trash_path(self, original_path: Path, batch_id: str) -> Path:
        """Compute the trash path for a file being 'deleted'.

        Files go to: <trash_dir>/<batch_id>/<original_filename>
        Handles name collisions by appending a counter.
        """
        base = self.trash_dir / batch_id
        dest = base / original_path.name
        # Handle name collisions (same filename from different directories)
        counter = 1
        while dest.exists():
            dest = base / f"{original_path.stem}_{counter}{original_path.suffix}"
            counter += 1
        return dest
