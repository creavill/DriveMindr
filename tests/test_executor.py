"""
Tests for the execution engine, undo system, and symlink manager.

Uses real filesystem operations in tmp directories — no mocking of file I/O.
Tests verify: moves, deletes-to-trash, archives, checksums, undo, dry-run,
symlinks, and the full executor pipeline.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from drivemindr.database import Database
from drivemindr.executor import ExecutionEngine, _categorize_destination, _compute_dest_path
from drivemindr.symlinks import AppMigrator, create_junction, is_junction, remove_junction
from drivemindr.undo import UndoManager, file_checksum, generate_batch_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


@pytest.fixture
def undo(db, tmp_path) -> UndoManager:
    trash = tmp_path / "trash"
    return UndoManager(db, trash_dir=trash)


def _insert_file(db: Database, path: str, name: str, ext: str, size: int = 1024) -> int:
    return db.upsert_file({
        "path": path,
        "name": name,
        "extension": ext,
        "size_bytes": size,
        "created": "2024-01-01T00:00:00",
        "modified": "2024-06-01T00:00:00",
        "accessed": "2024-12-01T00:00:00",
        "owner": "TestUser",
        "is_readonly": 0,
        "is_dir": 0,
        "parent_dir": str(Path(path).parent),
        "scan_id": "test001",
    })


def _classify(db: Database, file_id: int, action: str, confidence: float = 0.9) -> None:
    sql = """
    INSERT INTO classifications (file_id, action, confidence, reason, category, overridden)
    VALUES (?, ?, ?, 'test', 'test', 0)
    ON CONFLICT(file_id) DO UPDATE SET action=excluded.action, confidence=excluded.confidence
    """
    with db.transaction() as cur:
        cur.execute(sql, (file_id, action, confidence))


def _approve(db: Database, file_id: int, new_action: str | None = None) -> None:
    db.save_user_decision(file_id, "APPROVE", new_action)


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

class TestChecksum:

    def test_file_checksum(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        cs = file_checksum(f)
        assert cs is not None
        assert len(cs) == 64  # sha256 hex length

    def test_same_content_same_checksum(self, tmp_path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        assert file_checksum(f1) == file_checksum(f2)

    def test_different_content_different_checksum(self, tmp_path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert file_checksum(f1) != file_checksum(f2)

    def test_nonexistent_returns_none(self, tmp_path) -> None:
        assert file_checksum(tmp_path / "nope.txt") is None


# ---------------------------------------------------------------------------
# Batch ID
# ---------------------------------------------------------------------------

class TestBatchId:

    def test_format(self) -> None:
        bid = generate_batch_id()
        assert bid.startswith("batch_")
        assert len(bid) > 20

    def test_unique(self) -> None:
        ids = {generate_batch_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Undo Manager
# ---------------------------------------------------------------------------

class TestUndoManager:

    def test_log_action(self, undo: UndoManager) -> None:
        log_id = undo.log_action(
            file_id=None, action="MOVED",
            source_path="/src/a.txt", dest_path="/dst/a.txt",
            batch_id="batch_test",
        )
        assert log_id > 0

    def test_get_batch_actions(self, undo: UndoManager) -> None:
        bid = "batch_test_1"
        undo.log_action(None, "MOVED", "/src/a.txt", "/dst/a.txt", bid)
        undo.log_action(None, "DELETED", "/src/b.txt", "/trash/b.txt", bid)

        actions = undo.get_batch_actions(bid)
        assert len(actions) == 2
        # Should be in reverse order (for undo)
        assert actions[0]["action"] == "DELETED"
        assert actions[1]["action"] == "MOVED"

    def test_undo_move(self, undo: UndoManager, tmp_path) -> None:
        src = tmp_path / "original" / "file.txt"
        dst = tmp_path / "moved" / "file.txt"
        src.parent.mkdir(parents=True)
        dst.parent.mkdir(parents=True)
        dst.write_text("content")  # file is at dest (after a move)

        bid = "batch_undo_move"
        undo.log_action(None, "MOVED", str(src), str(dst), bid)

        result = undo.undo_batch(bid)
        assert result["undone"] == 1
        assert src.exists()
        assert not dst.exists()

    def test_undo_delete(self, undo: UndoManager, tmp_path) -> None:
        src = tmp_path / "original" / "deleted.txt"
        trash = tmp_path / "trash" / "deleted.txt"
        trash.parent.mkdir(parents=True)
        trash.write_text("restored content")

        bid = "batch_undo_delete"
        undo.log_action(None, "DELETED", str(src), str(trash), bid)

        result = undo.undo_batch(bid)
        assert result["undone"] == 1
        assert src.exists()
        assert src.read_text() == "restored content"

    def test_undo_archive(self, undo: UndoManager, tmp_path) -> None:
        archive = tmp_path / "archive" / "file.zip"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"fake zip")

        bid = "batch_undo_archive"
        undo.log_action(None, "ARCHIVED", "/src/file.txt", str(archive), bid)

        result = undo.undo_batch(bid)
        assert result["undone"] == 1
        assert not archive.exists()

    def test_undo_dry_run_changes_nothing(self, undo: UndoManager, tmp_path) -> None:
        dst = tmp_path / "moved" / "file.txt"
        dst.parent.mkdir(parents=True)
        dst.write_text("content")

        bid = "batch_dry"
        undo.log_action(None, "MOVED", "/src/file.txt", str(dst), bid)

        result = undo.undo_batch(bid, dry_run=True)
        assert result["undone"] == 1  # counted as "would undo"
        assert dst.exists()  # but file wasn't actually moved back

    def test_undo_missing_dest_skips(self, undo: UndoManager) -> None:
        bid = "batch_missing"
        undo.log_action(None, "MOVED", "/nope/src.txt", "/nope/dst.txt", bid)

        result = undo.undo_batch(bid)
        assert result["skipped"] == 1

    def test_get_recent_batches(self, undo: UndoManager) -> None:
        undo.log_action(None, "MOVED", "/a", "/b", "batch_a")
        undo.log_action(None, "MOVED", "/c", "/d", "batch_b")

        batches = undo.get_recent_batches()
        assert len(batches) == 2
        batch_ids = [b["batch_id"] for b in batches]
        assert "batch_b" in batch_ids

    def test_trash_path(self, undo: UndoManager) -> None:
        path = undo.get_trash_path(Path(r"C:\Users\test\file.tmp"), "batch_123")
        assert "batch_123" in str(path)
        assert "file.tmp" in str(path)


# ---------------------------------------------------------------------------
# Destination categorization
# ---------------------------------------------------------------------------

class TestCategorizeDestination:

    def test_document(self) -> None:
        assert _categorize_destination(r"C:\Users\doc.pdf", ".pdf") == "documents"

    def test_photo(self) -> None:
        assert _categorize_destination(r"C:\Users\photo.jpg", ".jpg") == "media_photos"

    def test_video(self) -> None:
        assert _categorize_destination(r"C:\Users\vid.mp4", ".mp4") == "media_videos"

    def test_music(self) -> None:
        assert _categorize_destination(r"C:\Users\song.mp3", ".mp3") == "media_music"

    def test_source_code(self) -> None:
        assert _categorize_destination(r"C:\Projects\app.py", ".py") == "projects"

    def test_path_hint_projects(self) -> None:
        assert _categorize_destination(r"C:\Users\Projects\data.bin", ".bin") == "projects"

    def test_unknown_defaults_to_documents(self) -> None:
        assert _categorize_destination(r"C:\Users\random.xyz", ".xyz") == "documents"


class TestComputeDestPath:

    def test_preserves_subfolder_structure(self) -> None:
        dest = _compute_dest_path(r"C:\Users\Conner\Documents\Work\report.pdf", "documents")
        assert str(dest).endswith("Work\\report.pdf") or str(dest).endswith("Work/report.pdf")

    def test_short_path_uses_filename(self) -> None:
        dest = _compute_dest_path(r"C:\file.txt", "documents")
        assert dest.name == "file.txt"


# ---------------------------------------------------------------------------
# Symlinks / Junctions
# ---------------------------------------------------------------------------

class TestSymlinks:

    def test_create_and_detect_junction(self, tmp_path) -> None:
        target = tmp_path / "app_data"
        target.mkdir()
        (target / "file.txt").write_text("app file")

        link = tmp_path / "app_link"
        assert create_junction(link, target)
        assert link.exists()
        assert (link / "file.txt").read_text() == "app file"

    def test_remove_junction(self, tmp_path) -> None:
        target = tmp_path / "target_dir"
        target.mkdir()
        link = tmp_path / "link_dir"
        create_junction(link, target)

        assert remove_junction(link)
        assert not link.exists()
        assert target.exists()  # target should NOT be removed


class TestAppMigrator:

    def test_migrate_app(self, undo: UndoManager, tmp_path) -> None:
        source = tmp_path / "source_app"
        source.mkdir()
        (source / "app.exe").write_text("binary")
        (source / "data").mkdir()
        (source / "data" / "config.ini").write_text("cfg")

        target_root = tmp_path / "Apps"
        migrator = AppMigrator(undo, target_root=target_root)

        result = migrator.migrate_app(source, file_id=None, batch_id="batch_mig")
        assert result["success"] is True
        assert result["junction_created"] is True

        # Original path should be a junction now
        assert source.exists()
        # Data should be at target
        assert (target_root / "source_app" / "app.exe").exists()
        # Can still access through junction
        assert (source / "app.exe").read_text() == "binary"

    def test_migrate_dry_run(self, undo: UndoManager, tmp_path) -> None:
        source = tmp_path / "app"
        source.mkdir()
        (source / "file.txt").write_text("content")

        target_root = tmp_path / "Apps"
        migrator = AppMigrator(undo, target_root=target_root)

        result = migrator.migrate_app(source, dry_run=True)
        assert result["success"] is True
        assert result.get("dry_run") is True
        # Source should be untouched
        assert source.exists()
        assert not target_root.exists()

    def test_migrate_nonexistent_source(self, undo: UndoManager, tmp_path) -> None:
        migrator = AppMigrator(undo, target_root=tmp_path / "Apps")
        result = migrator.migrate_app(tmp_path / "nope")
        assert result["success"] is False

    def test_migrate_target_exists(self, undo: UndoManager, tmp_path) -> None:
        source = tmp_path / "app"
        source.mkdir()
        target_root = tmp_path / "Apps"
        (target_root / "app").mkdir(parents=True)

        migrator = AppMigrator(undo, target_root=target_root)
        result = migrator.migrate_app(source)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Execution Engine — full pipeline
# ---------------------------------------------------------------------------

class TestExecutionEngine:

    def _setup_engine(
        self, db: Database, tmp_path: Path,
    ) -> tuple[ExecutionEngine, Path, Path]:
        """Create an engine with tmp-based directories."""
        trash = tmp_path / "trash"
        undo = UndoManager(db, trash_dir=trash)
        app_target = tmp_path / "Apps"
        migrator = AppMigrator(undo, target_root=app_target)
        engine = ExecutionEngine(
            db, undo=undo, app_migrator=migrator, trash_dir=trash,
        )
        return engine, trash, app_target

    def test_dry_run_makes_no_changes(self, db: Database, tmp_path) -> None:
        # Create a real file
        src = tmp_path / "source" / "junk.tmp"
        src.parent.mkdir(parents=True)
        src.write_text("junk content")

        fid = _insert_file(db, str(src), "junk.tmp", ".tmp", 100)
        _classify(db, fid, "DELETE_JUNK")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan(dry_run=True)

        assert summary["deleted"] == 1
        assert src.exists()  # file should NOT be touched

    def test_move_file(self, db: Database, tmp_path) -> None:
        src = tmp_path / "Users" / "test" / "Documents" / "report.pdf"
        src.parent.mkdir(parents=True)
        src.write_text("pdf content")

        fid = _insert_file(db, str(src), "report.pdf", ".pdf", 100)
        _classify(db, fid, "MOVE_DATA")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()

        assert summary["moved"] == 1
        assert not src.exists()  # original moved away

    def test_delete_to_trash(self, db: Database, tmp_path) -> None:
        src = tmp_path / "source" / "junk.tmp"
        src.parent.mkdir(parents=True)
        src.write_text("junk")

        fid = _insert_file(db, str(src), "junk.tmp", ".tmp", 100)
        _classify(db, fid, "DELETE_JUNK")
        _approve(db, fid)

        engine, trash, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()

        assert summary["deleted"] == 1
        assert not src.exists()
        # Should be in trash
        trash_files = list(trash.rglob("junk.tmp"))
        assert len(trash_files) == 1

    def test_archive_file(self, db: Database, tmp_path) -> None:
        src = tmp_path / "source" / "old_data.bin"
        src.parent.mkdir(parents=True)
        src.write_text("old data")

        fid = _insert_file(db, str(src), "old_data.bin", ".bin", 100)
        _classify(db, fid, "ARCHIVE")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()

        assert summary["archived"] == 1
        # Original still exists (archive is additive)
        assert src.exists()

    def test_undo_after_delete(self, db: Database, tmp_path) -> None:
        """Delete then undo should restore the file."""
        src = tmp_path / "source" / "important.txt"
        src.parent.mkdir(parents=True)
        src.write_text("important data")

        fid = _insert_file(db, str(src), "important.txt", ".txt", 100)
        _classify(db, fid, "DELETE_JUNK")
        _approve(db, fid)

        engine, trash, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()
        assert summary["deleted"] == 1
        assert not src.exists()

        # Now undo
        undo_mgr = UndoManager(db, trash_dir=trash)
        undo_result = undo_mgr.undo_batch(summary["batch_id"])
        assert undo_result["undone"] == 1
        assert src.exists()
        assert src.read_text() == "important data"

    def test_undo_after_move(self, db: Database, tmp_path) -> None:
        """Move then undo should restore the file to original location."""
        src = tmp_path / "Users" / "test" / "Documents" / "file.pdf"
        src.parent.mkdir(parents=True)
        src.write_text("pdf data")

        fid = _insert_file(db, str(src), "file.pdf", ".pdf", 100)
        _classify(db, fid, "MOVE_DATA")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()
        assert summary["moved"] == 1
        assert not src.exists()

        # Undo
        undo_mgr = engine.undo
        undo_result = undo_mgr.undo_batch(summary["batch_id"])
        assert undo_result["undone"] == 1
        assert src.exists()

    def test_no_approved_actions(self, db: Database, tmp_path) -> None:
        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()
        assert summary["moved"] == 0
        assert summary["deleted"] == 0
        assert summary["batch_id"] is None

    def test_skip_nonexistent_source(self, db: Database, tmp_path) -> None:
        fid = _insert_file(db, "/nonexistent/file.tmp", "file.tmp", ".tmp", 100)
        _classify(db, fid, "DELETE_JUNK")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()
        assert summary["skipped"] == 1

    def test_checksum_verified_on_move(self, db: Database, tmp_path) -> None:
        """Verify that files are checksummed before and after move."""
        src = tmp_path / "Users" / "test" / "Documents" / "data.csv"
        src.parent.mkdir(parents=True)
        src.write_text("a,b,c\n1,2,3")

        original_checksum = file_checksum(src)

        fid = _insert_file(db, str(src), "data.csv", ".csv", 100)
        _classify(db, fid, "MOVE_DATA")
        _approve(db, fid)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()
        assert summary["moved"] == 1
        assert summary["errors"] == 0

        # Check the logged checksum matches
        actions = engine.undo.get_batch_actions(summary["batch_id"])
        assert len(actions) == 1
        assert actions[0]["checksum_before"] == original_checksum
        assert actions[0]["checksum_after"] == original_checksum  # same content

    def test_multiple_operations_in_one_batch(self, db: Database, tmp_path) -> None:
        """Multiple file types processed in a single batch."""
        # File to delete
        junk = tmp_path / "src" / "junk.tmp"
        junk.parent.mkdir(parents=True)
        junk.write_text("junk")
        fid1 = _insert_file(db, str(junk), "junk.tmp", ".tmp")
        _classify(db, fid1, "DELETE_JUNK")
        _approve(db, fid1)

        # File to archive
        old = tmp_path / "src" / "old.bin"
        old.write_text("old data")
        fid2 = _insert_file(db, str(old), "old.bin", ".bin")
        _classify(db, fid2, "ARCHIVE")
        _approve(db, fid2)

        engine, _, _ = self._setup_engine(db, tmp_path)
        summary = engine.execute_plan()

        assert summary["deleted"] == 1
        assert summary["archived"] == 1
        assert summary["errors"] == 0
        assert summary["batch_id"] is not None
