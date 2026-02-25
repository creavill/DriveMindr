"""Tests for the SQLite database module."""

import pytest

from drivemindr.database import Database


@pytest.fixture
def db(tmp_path) -> Database:
    """Provide a clean in-memory-like database for each test."""
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


def _sample_file(path: str = r"C:\Users\test\file.txt", **overrides) -> dict:
    base = {
        "path": path,
        "name": "file.txt",
        "extension": ".txt",
        "size_bytes": 1024,
        "created": "2024-01-01T00:00:00",
        "modified": "2024-06-01T00:00:00",
        "accessed": "2024-12-01T00:00:00",
        "owner": "TestUser",
        "is_readonly": 0,
        "is_dir": 0,
        "parent_dir": r"C:\Users\test",
        "scan_id": "test001",
    }
    base.update(overrides)
    return base


class TestDatabaseConnect:

    def test_schema_created(self, db: Database) -> None:
        """All expected tables should exist after connect."""
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "files" in table_names
        assert "classifications" in table_names
        assert "user_decisions" in table_names
        assert "action_log" in table_names
        assert "installed_apps" in table_names
        assert "dir_sizes" in table_names


class TestFileOperations:

    def test_upsert_and_retrieve(self, db: Database) -> None:
        row_id = db.upsert_file(_sample_file())
        assert row_id is not None
        row = db.get_file_by_path(r"C:\Users\test\file.txt")
        assert row is not None
        assert row["name"] == "file.txt"
        assert row["size_bytes"] == 1024

    def test_upsert_updates_existing(self, db: Database) -> None:
        db.upsert_file(_sample_file(size_bytes=100))
        db.upsert_file(_sample_file(size_bytes=200))
        assert db.file_count() == 1
        row = db.get_file_by_path(r"C:\Users\test\file.txt")
        assert row["size_bytes"] == 200

    def test_bulk_upsert(self, db: Database) -> None:
        records = [
            _sample_file(path=f"C:\\file{i}.txt", name=f"file{i}.txt")
            for i in range(100)
        ]
        count = db.bulk_upsert_files(records)
        assert count == 100
        assert db.file_count() == 100

    def test_top_largest(self, db: Database) -> None:
        for i in range(5):
            db.upsert_file(_sample_file(
                path=f"C:\\file{i}.dat",
                name=f"file{i}.dat",
                size_bytes=(i + 1) * 1000,
            ))
        top = db.get_top_largest(3)
        assert len(top) == 3
        assert top[0]["size_bytes"] == 5000  # largest first

    def test_total_size(self, db: Database) -> None:
        db.upsert_file(_sample_file(path="C:\\a.txt", size_bytes=100))
        db.upsert_file(_sample_file(path="C:\\b.txt", size_bytes=200))
        assert db.total_size() == 300

    def test_get_files_by_extension(self, db: Database) -> None:
        db.upsert_file(_sample_file(path="C:\\a.txt", extension=".txt"))
        db.upsert_file(_sample_file(path="C:\\b.py", extension=".py", name="b.py"))
        rows = db.get_files(extension=".txt")
        assert len(rows) == 1
        assert rows[0]["extension"] == ".txt"

    def test_get_files_by_min_size(self, db: Database) -> None:
        db.upsert_file(_sample_file(path="C:\\small.txt", size_bytes=10))
        db.upsert_file(_sample_file(path="C:\\big.txt", size_bytes=10000))
        rows = db.get_files(min_size=1000)
        assert len(rows) == 1
        assert rows[0]["size_bytes"] == 10000


class TestDirSizes:

    def test_upsert_and_retrieve(self, db: Database) -> None:
        db.upsert_dir_size(r"C:\Users\test", 50000, 25, "scan01")
        rows = db.get_dir_sizes()
        assert len(rows) == 1
        assert rows[0]["total_bytes"] == 50000
        assert rows[0]["file_count"] == 25


class TestInstalledApps:

    def test_upsert_and_retrieve(self, db: Database) -> None:
        db.upsert_installed_app({
            "display_name": "TestApp",
            "install_location": r"C:\Program Files\TestApp",
            "publisher": "TestCorp",
            "install_date": "20240101",
            "estimated_size": 1048576,
            "uninstall_string": "uninstall.exe",
            "registry_key": r"SOFTWARE\TestApp",
        })
        apps = db.get_installed_apps()
        assert len(apps) == 1
        assert apps[0]["display_name"] == "TestApp"


class TestTransaction:

    def test_rollback_on_error(self, db: Database) -> None:
        """Transaction should roll back if an error occurs."""
        db.upsert_file(_sample_file())
        try:
            with db.transaction() as cur:
                cur.execute("DELETE FROM files")
                raise ValueError("simulated error")
        except ValueError:
            pass
        # File should still exist due to rollback
        assert db.file_count() == 1
