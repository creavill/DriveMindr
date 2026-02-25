"""Tests for the file scanner module."""

import os

import pytest

from drivemindr.database import Database
from drivemindr.scanner import FileScanner, _is_under_protected_path, _timestamp


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


@pytest.fixture
def scan_tree(tmp_path):
    """Create a dedicated subdirectory tree for scanning (avoids pytest artifacts)."""
    root = tmp_path / "scan_root"
    root.mkdir()

    (root / "file1.txt").write_text("hello world")
    (root / "file2.py").write_text("print('hi')")
    (root / "large.bin").write_bytes(b"\x00" * 4096)

    sub = root / "subdir"
    sub.mkdir()
    (sub / "nested.md").write_text("# Readme")

    deep = sub / "deep"
    deep.mkdir()
    (deep / "data.csv").write_text("a,b,c\n1,2,3")

    return root


class TestTimestamp:

    def test_valid_epoch(self) -> None:
        result = _timestamp(0.0)
        assert result is not None
        assert "1970" in result or "1969" in result  # timezone dependent

    def test_none_returns_none(self) -> None:
        assert _timestamp(None) is None


class TestProtectedPathCheck:

    def test_windows_system_protected(self) -> None:
        assert _is_under_protected_path(r"C:\Windows\System32\notepad.exe") is True

    def test_boot_protected(self) -> None:
        assert _is_under_protected_path(r"C:\Boot\BCD") is True

    def test_user_dir_not_protected(self) -> None:
        assert _is_under_protected_path(r"C:\Users\Conner\file.txt") is False


class TestFileScanner:

    def test_scan_counts_files(self, db: Database, scan_tree) -> None:
        scanner = FileScanner(db)
        summary = scanner.scan(str(scan_tree))
        assert summary["files"] == 5  # file1.txt, file2.py, large.bin, nested.md, data.csv
        assert summary["dirs"] == 2  # subdir, deep
        assert summary["errors"] == 0
        assert summary["total_bytes"] > 0

    def test_scan_stores_in_database(self, db: Database, scan_tree) -> None:
        scanner = FileScanner(db)
        scanner.scan(str(scan_tree))
        assert db.file_count() > 0

    def test_scan_nonexistent_raises(self, db: Database) -> None:
        scanner = FileScanner(db)
        with pytest.raises(FileNotFoundError):
            scanner.scan("/nonexistent/path")

    def test_scan_captures_extensions(self, db: Database, scan_tree) -> None:
        scanner = FileScanner(db)
        scanner.scan(str(scan_tree))
        py_files = db.get_files(extension=".py")
        assert len(py_files) == 1
        assert py_files[0]["name"] == "file2.py"

    def test_scan_captures_sizes(self, db: Database, scan_tree) -> None:
        scanner = FileScanner(db)
        scanner.scan(str(scan_tree))
        large = db.get_files(min_size=4096)
        assert len(large) == 1
        assert large[0]["name"] == "large.bin"

    def test_scan_populates_dir_sizes(self, db: Database, scan_tree) -> None:
        scanner = FileScanner(db)
        scanner.scan(str(scan_tree))
        dirs = db.get_dir_sizes()
        assert len(dirs) > 0

    def test_progress_callback_called(self, db: Database, scan_tree) -> None:
        calls = []
        scanner = FileScanner(db)
        scanner.scan(str(scan_tree), progress_callback=lambda s, e: calls.append((s, e)))
        # Should be called at least once
        assert len(calls) >= 1

    def test_scan_id_unique(self, db: Database, scan_tree) -> None:
        s1 = FileScanner(db)
        s2 = FileScanner(db)
        assert s1.scan_id != s2.scan_id
