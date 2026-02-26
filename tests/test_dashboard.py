"""
Tests for dashboard-related database queries and user decisions.

Tests the DB query layer that the Streamlit dashboard depends on.
No Streamlit import needed — we test the data layer directly.
"""

from __future__ import annotations

import pytest

from drivemindr.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


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
        "parent_dir": "C:\\Users\\test",
        "scan_id": "test001",
    })


def _classify_file(db: Database, file_id: int, action: str, confidence: float = 0.9) -> None:
    sql = """
    INSERT INTO classifications (file_id, action, confidence, reason, category, overridden, override_reason)
    VALUES (?, ?, ?, 'test reason', 'test', 0, NULL)
    ON CONFLICT(file_id) DO UPDATE SET
        action=excluded.action, confidence=excluded.confidence
    """
    with db.transaction() as cur:
        cur.execute(sql, (file_id, action, confidence))


def _setup_classified_files(db: Database) -> dict[str, int]:
    """Insert a set of files with classifications. Returns path→id mapping."""
    files = {
        r"C:\temp\junk.tmp": ("junk.tmp", ".tmp", 5000, "DELETE_JUNK"),
        r"C:\old\unused.exe": ("unused.exe", ".exe", 50000, "DELETE_UNUSED"),
        r"C:\Users\data.bin": ("data.bin", ".bin", 100000, "MOVE_DATA"),
        r"C:\Apps\steam.exe": ("steam.exe", ".exe", 200000, "MOVE_APP"),
        r"C:\docs\report.pdf": ("report.pdf", ".pdf", 3000, "KEEP"),
        r"C:\old\archive.zip": ("archive.zip", ".zip", 80000, "ARCHIVE"),
    }
    ids = {}
    for path, (name, ext, size, action) in files.items():
        fid = _insert_file(db, path, name, ext, size)
        _classify_file(db, fid, action)
        ids[path] = fid
    return ids


# ---------------------------------------------------------------------------
# Classification summary queries
# ---------------------------------------------------------------------------

class TestClassificationSummary:

    def test_empty_db(self, db: Database) -> None:
        summary = db.get_classification_summary()
        assert summary == {}

    def test_summary_counts_and_bytes(self, db: Database) -> None:
        _setup_classified_files(db)
        summary = db.get_classification_summary()

        assert "DELETE_JUNK" in summary
        assert summary["DELETE_JUNK"]["count"] == 1
        assert summary["DELETE_JUNK"]["bytes"] == 5000

        assert "MOVE_DATA" in summary
        assert summary["MOVE_DATA"]["count"] == 1
        assert summary["MOVE_DATA"]["bytes"] == 100000

    def test_summary_includes_all_actions(self, db: Database) -> None:
        _setup_classified_files(db)
        summary = db.get_classification_summary()
        assert len(summary) == 6  # 6 distinct actions


class TestFilesbyAction:

    def test_returns_files_for_action(self, db: Database) -> None:
        _setup_classified_files(db)
        files = db.get_files_by_action("DELETE_JUNK")
        assert len(files) == 1
        assert files[0]["name"] == "junk.tmp"

    def test_includes_classification_data(self, db: Database) -> None:
        _setup_classified_files(db)
        files = db.get_files_by_action("MOVE_DATA")
        assert len(files) == 1
        f = files[0]
        assert f["ai_action"] == "MOVE_DATA"
        assert f["confidence"] == 0.9
        assert f["reason"] == "test reason"

    def test_empty_action_returns_empty(self, db: Database) -> None:
        _setup_classified_files(db)
        files = db.get_files_by_action("NONEXISTENT")
        assert files == []

    def test_includes_user_decision_if_exists(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        junk_id = ids[r"C:\temp\junk.tmp"]
        db.save_user_decision(junk_id, "APPROVE")

        files = db.get_files_by_action("DELETE_JUNK")
        assert files[0]["decision"] == "APPROVE"


# ---------------------------------------------------------------------------
# User decisions
# ---------------------------------------------------------------------------

class TestUserDecisions:

    def test_save_and_retrieve(self, db: Database) -> None:
        fid = _insert_file(db, r"C:\test.txt", "test.txt", ".txt")
        _classify_file(db, fid, "DELETE_JUNK")

        db.save_user_decision(fid, "APPROVE")
        files = db.get_files_by_action("DELETE_JUNK")
        assert files[0]["decision"] == "APPROVE"

    def test_save_with_new_action(self, db: Database) -> None:
        fid = _insert_file(db, r"C:\test.txt", "test.txt", ".txt")
        _classify_file(db, fid, "DELETE_JUNK")

        db.save_user_decision(fid, "CHANGE", "KEEP")
        files = db.get_files_by_action("DELETE_JUNK")
        assert files[0]["decision"] == "CHANGE"
        assert files[0]["new_action"] == "KEEP"

    def test_upsert_overwrites_previous(self, db: Database) -> None:
        fid = _insert_file(db, r"C:\test.txt", "test.txt", ".txt")
        _classify_file(db, fid, "DELETE_JUNK")

        db.save_user_decision(fid, "APPROVE")
        db.save_user_decision(fid, "REJECT")  # override

        files = db.get_files_by_action("DELETE_JUNK")
        assert files[0]["decision"] == "REJECT"

    def test_batch_decisions(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        all_ids = list(ids.values())

        count = db.save_batch_decisions(all_ids, "APPROVE")
        assert count == len(all_ids)

        stats = db.get_review_stats()
        assert stats["approved"] == len(all_ids)


# ---------------------------------------------------------------------------
# Review stats
# ---------------------------------------------------------------------------

class TestReviewStats:

    def test_empty_db(self, db: Database) -> None:
        stats = db.get_review_stats()
        assert stats["classified"] == 0
        assert stats["reviewed"] == 0
        assert stats["pending"] == 0

    def test_with_classified_files(self, db: Database) -> None:
        _setup_classified_files(db)
        stats = db.get_review_stats()
        assert stats["classified"] == 6
        assert stats["pending"] == 6
        assert stats["reviewed"] == 0

    def test_after_reviews(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        junk_id = ids[r"C:\temp\junk.tmp"]
        data_id = ids[r"C:\Users\data.bin"]

        db.save_user_decision(junk_id, "APPROVE")
        db.save_user_decision(data_id, "REJECT")

        stats = db.get_review_stats()
        assert stats["classified"] == 6
        assert stats["reviewed"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 4


# ---------------------------------------------------------------------------
# Approved actions (execution plan)
# ---------------------------------------------------------------------------

class TestApprovedActions:

    def test_empty_when_no_approvals(self, db: Database) -> None:
        _setup_classified_files(db)
        approved = db.get_approved_actions()
        assert approved == []

    def test_returns_only_approved(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        junk_id = ids[r"C:\temp\junk.tmp"]
        data_id = ids[r"C:\Users\data.bin"]
        keep_id = ids[r"C:\docs\report.pdf"]

        db.save_user_decision(junk_id, "APPROVE")
        db.save_user_decision(data_id, "APPROVE")
        db.save_user_decision(keep_id, "REJECT")

        approved = db.get_approved_actions()
        assert len(approved) == 2
        paths = {a["path"] for a in approved}
        assert r"C:\temp\junk.tmp" in paths
        assert r"C:\Users\data.bin" in paths

    def test_uses_changed_action_when_present(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        junk_id = ids[r"C:\temp\junk.tmp"]

        # User changes DELETE_JUNK to ARCHIVE and approves
        db.save_user_decision(junk_id, "APPROVE", "ARCHIVE")

        approved = db.get_approved_actions()
        assert len(approved) == 1
        assert approved[0]["final_action"] == "ARCHIVE"


# ---------------------------------------------------------------------------
# Space recovery estimate
# ---------------------------------------------------------------------------

class TestSpaceRecovery:

    def test_empty_db(self, db: Database) -> None:
        recovery = db.get_space_recovery_estimate()
        assert recovery == {}

    def test_sums_delete_and_archive(self, db: Database) -> None:
        _setup_classified_files(db)
        recovery = db.get_space_recovery_estimate()

        assert recovery["DELETE_JUNK"] == 5000
        assert recovery["DELETE_UNUSED"] == 50000
        assert recovery["ARCHIVE"] == 80000
        # KEEP and MOVE should not appear
        assert "KEEP" not in recovery
        assert "MOVE_DATA" not in recovery


# ---------------------------------------------------------------------------
# Extension breakdown
# ---------------------------------------------------------------------------

class TestExtensionBreakdown:

    def test_returns_extension_groups(self, db: Database) -> None:
        _insert_file(db, r"C:\a.txt", "a.txt", ".txt", 1000)
        _insert_file(db, r"C:\b.txt", "b.txt", ".txt", 2000)
        _insert_file(db, r"C:\c.pdf", "c.pdf", ".pdf", 5000)

        data = db.get_extension_breakdown()
        assert len(data) == 2

        # .pdf should be first (5000 > 3000)
        assert data[0]["extension"] == ".pdf"
        assert data[0]["total_bytes"] == 5000
        assert data[1]["extension"] == ".txt"
        assert data[1]["file_count"] == 2
        assert data[1]["total_bytes"] == 3000


# ---------------------------------------------------------------------------
# Unreviewed files
# ---------------------------------------------------------------------------

class TestUnreviewedFiles:

    def test_returns_all_classified_when_none_reviewed(self, db: Database) -> None:
        _setup_classified_files(db)
        unreviewed = db.get_unreviewed_files()
        assert len(unreviewed) == 6

    def test_excludes_reviewed(self, db: Database) -> None:
        ids = _setup_classified_files(db)
        db.save_user_decision(ids[r"C:\temp\junk.tmp"], "APPROVE")
        db.save_user_decision(ids[r"C:\Users\data.bin"], "REJECT")

        unreviewed = db.get_unreviewed_files()
        assert len(unreviewed) == 4

    def test_orders_deletes_first(self, db: Database) -> None:
        _setup_classified_files(db)
        unreviewed = db.get_unreviewed_files()
        # DELETE_JUNK and DELETE_UNUSED should come before MOVE/KEEP
        actions = [f["ai_action"] for f in unreviewed]
        delete_indices = [i for i, a in enumerate(actions) if a.startswith("DELETE")]
        other_indices = [i for i, a in enumerate(actions) if not a.startswith("DELETE")]
        if delete_indices and other_indices:
            assert max(delete_indices) < min(other_indices)
