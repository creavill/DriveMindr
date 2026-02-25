"""
DriveMindr SQLite database — file index, classifications, action log, undo history.

Single-file database stored locally. No network. No cloud sync.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from drivemindr.config import DEFAULT_DB_NAME

logger = logging.getLogger("drivemindr.database")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
-- Raw file metadata populated by the scanner
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    extension   TEXT,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    created     TEXT,
    modified    TEXT,
    accessed    TEXT,
    owner       TEXT,
    is_readonly INTEGER NOT NULL DEFAULT 0,
    is_dir      INTEGER NOT NULL DEFAULT 0,
    parent_dir  TEXT,
    scan_id     TEXT,
    scanned_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- AI classifications (populated by classifier.py in Phase 2)
CREATE TABLE IF NOT EXISTS classifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    action      TEXT    NOT NULL,   -- KEEP | MOVE_APP | MOVE_DATA | DELETE_JUNK | DELETE_UNUSED | ARCHIVE
    confidence  REAL    NOT NULL DEFAULT 0.0,
    reason      TEXT,
    category    TEXT,
    overridden  INTEGER NOT NULL DEFAULT 0,  -- 1 if safety engine overrode AI
    override_reason TEXT,
    classified_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE(file_id)
);

-- User decisions from the review dashboard (Phase 3)
CREATE TABLE IF NOT EXISTS user_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    decision    TEXT    NOT NULL,   -- APPROVE | REJECT | CHANGE | PROTECT
    new_action  TEXT,               -- if decision=CHANGE, what the user picked
    decided_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE(file_id)
);

-- Execution log — every action taken (Phase 4)
CREATE TABLE IF NOT EXISTS action_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER REFERENCES files(id),
    action          TEXT NOT NULL,       -- MOVED | DELETED | ARCHIVED | SYMLINKED
    source_path     TEXT NOT NULL,
    dest_path       TEXT,
    checksum_before TEXT,
    checksum_after  TEXT,
    batch_id        TEXT,                -- groups actions for batch undo
    executed_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    undone          INTEGER NOT NULL DEFAULT 0
);

-- Installed applications (from Windows Registry scan)
CREATE TABLE IF NOT EXISTS installed_apps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name    TEXT,
    install_location TEXT,
    publisher       TEXT,
    install_date    TEXT,
    estimated_size  INTEGER,
    uninstall_string TEXT,
    registry_key    TEXT UNIQUE,
    scanned_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Directory-level size aggregations for the dashboard
CREATE TABLE IF NOT EXISTS dir_sizes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    file_count  INTEGER NOT NULL DEFAULT 0,
    scan_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_path       ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_extension   ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_parent      ON files(parent_dir);
CREATE INDEX IF NOT EXISTS idx_files_size        ON files(size_bytes DESC);
CREATE INDEX IF NOT EXISTS idx_classifications_action ON classifications(action);
CREATE INDEX IF NOT EXISTS idx_action_log_batch  ON action_log(batch_id);
"""


class Database:
    """Thin wrapper around a local SQLite database for DriveMindr."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path(DEFAULT_DB_NAME)
        self._conn: sqlite3.Connection | None = None
        logger.debug("Database configured at %s", self.db_path)

    # -- connection management ------------------------------------------------

    def connect(self) -> None:
        """Open (or create) the database and ensure the schema exists."""
        logger.info("Opening database: %s", self.db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.debug("Schema initialized")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected — call .connect() first")
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager that commits on success or rolls back on error."""
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            logger.exception("Transaction rolled back")
            raise

    # -- file operations ------------------------------------------------------

    def upsert_file(self, metadata: dict[str, Any]) -> int:
        """Insert or update a file record. Returns the row id."""
        sql = """
        INSERT INTO files (path, name, extension, size_bytes, created, modified,
                           accessed, owner, is_readonly, is_dir, parent_dir, scan_id)
        VALUES (:path, :name, :extension, :size_bytes, :created, :modified,
                :accessed, :owner, :is_readonly, :is_dir, :parent_dir, :scan_id)
        ON CONFLICT(path) DO UPDATE SET
            name=excluded.name, extension=excluded.extension,
            size_bytes=excluded.size_bytes, created=excluded.created,
            modified=excluded.modified, accessed=excluded.accessed,
            owner=excluded.owner, is_readonly=excluded.is_readonly,
            is_dir=excluded.is_dir, parent_dir=excluded.parent_dir,
            scan_id=excluded.scan_id, scanned_at=datetime('now','localtime')
        """
        with self.transaction() as cur:
            cur.execute(sql, metadata)
            row_id = cur.lastrowid
        logger.debug("Upserted file id=%s path=%s", row_id, metadata.get("path"))
        return row_id  # type: ignore[return-value]

    def bulk_upsert_files(self, records: list[dict[str, Any]]) -> int:
        """Bulk insert/update file records. Returns count inserted."""
        sql = """
        INSERT INTO files (path, name, extension, size_bytes, created, modified,
                           accessed, owner, is_readonly, is_dir, parent_dir, scan_id)
        VALUES (:path, :name, :extension, :size_bytes, :created, :modified,
                :accessed, :owner, :is_readonly, :is_dir, :parent_dir, :scan_id)
        ON CONFLICT(path) DO UPDATE SET
            name=excluded.name, extension=excluded.extension,
            size_bytes=excluded.size_bytes, created=excluded.created,
            modified=excluded.modified, accessed=excluded.accessed,
            owner=excluded.owner, is_readonly=excluded.is_readonly,
            is_dir=excluded.is_dir, parent_dir=excluded.parent_dir,
            scan_id=excluded.scan_id, scanned_at=datetime('now','localtime')
        """
        with self.transaction() as cur:
            cur.executemany(sql, records)
            count = cur.rowcount
        logger.info("Bulk upserted %d file records", count)
        return count

    def upsert_dir_size(self, path: str, total_bytes: int, file_count: int, scan_id: str) -> None:
        sql = """
        INSERT INTO dir_sizes (path, total_bytes, file_count, scan_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            total_bytes=excluded.total_bytes,
            file_count=excluded.file_count,
            scan_id=excluded.scan_id
        """
        with self.transaction() as cur:
            cur.execute(sql, (path, total_bytes, file_count, scan_id))

    def upsert_installed_app(self, app: dict[str, Any]) -> None:
        sql = """
        INSERT INTO installed_apps
            (display_name, install_location, publisher, install_date,
             estimated_size, uninstall_string, registry_key)
        VALUES
            (:display_name, :install_location, :publisher, :install_date,
             :estimated_size, :uninstall_string, :registry_key)
        ON CONFLICT(registry_key) DO UPDATE SET
            display_name=excluded.display_name,
            install_location=excluded.install_location,
            publisher=excluded.publisher,
            install_date=excluded.install_date,
            estimated_size=excluded.estimated_size,
            uninstall_string=excluded.uninstall_string
        """
        with self.transaction() as cur:
            cur.execute(sql, app)

    # -- query helpers --------------------------------------------------------

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM files WHERE path = ?", (path,))
        return cur.fetchone()

    def get_files(
        self,
        *,
        extension: str | None = None,
        min_size: int | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if extension:
            clauses.append("extension = ?")
            params.append(extension)
        if min_size is not None:
            clauses.append("size_bytes >= ?")
            params.append(min_size)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM files {where} ORDER BY size_bytes DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return self.conn.execute(sql, params).fetchall()

    def get_top_largest(self, n: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM files ORDER BY size_bytes DESC LIMIT ?", (n,)
        ).fetchall()

    def get_dir_sizes(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM dir_sizes ORDER BY total_bytes DESC LIMIT ?", (limit,)
        ).fetchall()

    def get_installed_apps(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM installed_apps ORDER BY display_name"
        ).fetchall()

    def file_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return row[0] if row else 0

    def total_size(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(size_bytes),0) FROM files").fetchone()
        return row[0] if row else 0
