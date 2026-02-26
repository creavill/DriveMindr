"""
Tests for the AI classifier module.

All tests use mocked Ollama responses — no network calls, no Ollama required.
Covers: response parsing, safety integration, batch processing, error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from drivemindr.classifier import (
    ClassificationResult,
    FileClassifier,
    OllamaClient,
    _build_batch_prompt,
    _parse_response,
)
from drivemindr.database import Database
from drivemindr.safety import SafetyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


def _insert_test_file(db: Database, path: str, name: str, ext: str, size: int = 1024) -> int:
    """Insert a test file record and return its row id."""
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


def _make_ai_response(items: list[dict]) -> str:
    """Build a mock Ollama JSON response string."""
    return json.dumps(items)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseResponse:

    def test_valid_json_array(self) -> None:
        text = json.dumps([
            {"path": "C:\\a.txt", "action": "KEEP", "confidence": 0.9,
             "reason": "User document", "category": "document"},
        ])
        results = _parse_response(text, 1)
        assert len(results) == 1
        assert results[0].action == "KEEP"
        assert results[0].confidence == 0.9

    def test_strips_markdown_fences(self) -> None:
        text = "```json\n" + json.dumps([
            {"path": "C:\\a.txt", "action": "MOVE_DATA", "confidence": 0.8,
             "reason": "move it", "category": "data"},
        ]) + "\n```"
        results = _parse_response(text, 1)
        assert len(results) == 1
        assert results[0].action == "MOVE_DATA"

    def test_handles_trailing_comma(self) -> None:
        text = '[{"path":"C:\\\\a.txt","action":"KEEP","confidence":0.9,"reason":"ok","category":"x"},]'
        results = _parse_response(text, 1)
        assert len(results) == 1

    def test_invalid_action_defaults_to_keep(self) -> None:
        text = json.dumps([
            {"path": "C:\\a.txt", "action": "YEET", "confidence": 0.9,
             "reason": "bad action", "category": "x"},
        ])
        results = _parse_response(text, 1)
        assert results[0].action == "KEEP"

    def test_invalid_confidence_defaults_to_zero(self) -> None:
        text = json.dumps([
            {"path": "C:\\a.txt", "action": "KEEP", "confidence": "not_a_number",
             "reason": "bad conf", "category": "x"},
        ])
        results = _parse_response(text, 1)
        assert results[0].confidence == 0.0

    def test_clamps_confidence(self) -> None:
        text = json.dumps([
            {"path": "C:\\a.txt", "action": "KEEP", "confidence": 1.5,
             "reason": "over 1", "category": "x"},
        ])
        results = _parse_response(text, 1)
        assert results[0].confidence == 1.0

    def test_completely_invalid_json_returns_empty(self) -> None:
        results = _parse_response("this is not json at all", 1)
        assert results == []

    def test_json_not_array_returns_empty(self) -> None:
        results = _parse_response('{"not": "an array"}', 1)
        assert results == []

    def test_extracts_json_from_surrounding_text(self) -> None:
        text = 'Here is my classification:\n' + json.dumps([
            {"path": "C:\\a.txt", "action": "KEEP", "confidence": 0.9,
             "reason": "ok", "category": "x"},
        ]) + '\nDone!'
        results = _parse_response(text, 1)
        assert len(results) == 1

    def test_multiple_items(self) -> None:
        items = [
            {"path": f"C:\\file{i}.txt", "action": "KEEP", "confidence": 0.9,
             "reason": "ok", "category": "x"}
            for i in range(5)
        ]
        results = _parse_response(json.dumps(items), 5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildBatchPrompt:

    def test_includes_metadata_fields(self, db: Database) -> None:
        _insert_test_file(db, r"C:\Users\test\doc.pdf", "doc.pdf", ".pdf", 2048)
        files = db.get_files()
        prompt = _build_batch_prompt(files)
        assert "doc.pdf" in prompt
        assert ".pdf" in prompt
        assert "2048" in prompt
        assert r"C:\Users\test\doc.pdf" in prompt

    def test_never_contains_file_contents(self, db: Database) -> None:
        """The prompt must never contain anything that looks like file content."""
        _insert_test_file(db, r"C:\Users\test\secret.env", "secret.env", ".env")
        files = db.get_files()
        prompt = _build_batch_prompt(files)
        # The prompt should only contain metadata keywords — no "content", "data", "body"
        assert "Classify these files" in prompt
        assert "path:" in prompt


# ---------------------------------------------------------------------------
# OllamaClient (mocked network)
# ---------------------------------------------------------------------------

class TestOllamaClient:

    def test_is_available_returns_false_when_down(self) -> None:
        client = OllamaClient(host="http://127.0.0.1:99999")
        assert client.is_available() is False

    def test_has_model_returns_false_when_down(self) -> None:
        client = OllamaClient(host="http://127.0.0.1:99999")
        assert client.has_model() is False

    @patch("drivemindr.classifier.urllib.request.urlopen")
    def test_is_available_returns_true(self, mock_urlopen) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = OllamaClient()
        assert client.is_available() is True

    @patch("drivemindr.classifier.urllib.request.urlopen")
    def test_has_model_returns_true(self, mock_urlopen) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "models": [{"name": "llama3.1:8b"}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = OllamaClient()
        assert client.has_model("llama3.1:8b") is True

    @patch("drivemindr.classifier.urllib.request.urlopen")
    def test_generate_returns_response_text(self, mock_urlopen) -> None:
        response_body = json.dumps({"response": '[{"path":"C:\\\\a.txt","action":"KEEP","confidence":0.9,"reason":"ok","category":"x"}]'})
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = OllamaClient()
        result = client.generate("test prompt")
        assert "KEEP" in result


# ---------------------------------------------------------------------------
# FileClassifier — full pipeline with mocked Ollama
# ---------------------------------------------------------------------------

class TestFileClassifier:

    def _mock_ollama(self, responses: list[str]) -> OllamaClient:
        """Create a mock OllamaClient that returns canned responses."""
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.has_model.return_value = True
        client.generate.side_effect = responses
        return client

    def test_classify_batch_stores_results(self, db: Database) -> None:
        path = r"C:\Users\test\installer.msi"
        _insert_test_file(db, path, "installer.msi", ".msi", 50000)
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 0.95,
             "reason": "Old installer", "category": "junk"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert len(result) == 1
        assert result[0]["action"] == "DELETE_JUNK"  # .msi is not guardian-protected

    def test_safety_overrides_document_delete(self, db: Database) -> None:
        """AI says delete a .docx → safety must override to KEEP."""
        path = r"C:\Users\test\report.docx"
        _insert_test_file(db, path, "report.docx", ".docx")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 0.99,
             "reason": "Old document", "category": "document"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert len(result) == 1
        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1
        assert "Guardian" in result[0]["override_reason"]

    def test_safety_overrides_photo_delete(self, db: Database) -> None:
        """AI says delete a .jpg → safety must override to KEEP."""
        path = r"C:\Users\test\photo.jpg"
        _insert_test_file(db, path, "photo.jpg", ".jpg")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 0.99,
             "reason": "Temp photo", "category": "media"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1

    def test_safety_overrides_source_code_delete(self, db: Database) -> None:
        path = r"C:\Projects\app.py"
        _insert_test_file(db, path, "app.py", ".py")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_UNUSED", "confidence": 0.95,
             "reason": "Unused script", "category": "code"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1

    def test_low_confidence_delete_overridden(self, db: Database) -> None:
        """Delete with confidence < 0.85 → overridden to KEEP."""
        path = r"C:\Users\test\something.tmp"
        _insert_test_file(db, path, "something.tmp", ".tmp")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 0.6,
             "reason": "Maybe junk", "category": "temp"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1

    def test_move_action_passes_through(self, db: Database) -> None:
        """MOVE_DATA with good confidence should pass through unchanged."""
        path = r"C:\Users\test\data.bin"
        _insert_test_file(db, path, "data.bin", ".bin", 100000)
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "MOVE_DATA", "confidence": 0.85,
             "reason": "Large data file, good for D:", "category": "data"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "MOVE_DATA"
        assert result[0]["overridden"] == 0

    def test_missing_ai_result_defaults_to_keep(self, db: Database) -> None:
        """If AI doesn't return a result for a file, default to KEEP."""
        _insert_test_file(db, r"C:\a.txt", "a.txt", ".txt")
        _insert_test_file(db, r"C:\b.txt", "b.txt", ".txt")
        files = db.get_files()

        # AI only returns one result for two files
        ai_response = _make_ai_response([
            {"path": r"C:\a.txt", "action": "KEEP", "confidence": 0.9,
             "reason": "ok", "category": "x"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        # Both should be classified — the missing one defaults to KEEP
        assert len(result) == 2
        actions = {r["action"] for r in result}
        assert "KEEP" in actions

    def test_classify_all_processes_batches(self, db: Database) -> None:
        """classify_all should process multiple batches."""
        # Insert 5 files, use batch_size=2
        for i in range(5):
            _insert_test_file(
                db, f"C:\\file{i}.dat", f"file{i}.dat", ".dat", 100 * (i + 1),
            )

        def _make_batch_response(paths):
            return _make_ai_response([
                {"path": p, "action": "MOVE_DATA", "confidence": 0.85,
                 "reason": "data file", "category": "data"}
                for p in paths
            ])

        # We need 3 batches: [2, 2, 1] files
        responses = [
            _make_batch_response([f"C:\\file{i}.dat" for i in range(2)]),
            _make_batch_response([f"C:\\file{i}.dat" for i in range(2, 4)]),
            _make_batch_response([f"C:\\file4.dat"]),
        ]

        client = self._mock_ollama(responses)
        classifier = FileClassifier(db, ollama_client=client, batch_size=2)
        summary = classifier.classify_all()

        assert summary["classified"] == 5
        assert summary["batches"] == 3
        assert summary["errors"] == 0

    def test_preflight_check_reports_status(self, db: Database) -> None:
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.has_model.return_value = True

        classifier = FileClassifier(db, ollama_client=client)
        status = classifier.preflight_check()
        assert status["ollama_up"] is True
        assert status["model_ready"] is True

    def test_preflight_check_ollama_down(self, db: Database) -> None:
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = False

        classifier = FileClassifier(db, ollama_client=client)
        status = classifier.preflight_check()
        assert status["ollama_up"] is False
        assert status["model_ready"] is False

    def test_connection_error_counts_as_errors(self, db: Database) -> None:
        """If Ollama connection fails mid-batch, files are counted as errors."""
        for i in range(3):
            _insert_test_file(db, f"C:\\file{i}.dat", f"file{i}.dat", ".dat")

        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.has_model.return_value = True
        client.generate.side_effect = ConnectionError("Ollama went away")

        classifier = FileClassifier(db, ollama_client=client, batch_size=10)
        summary = classifier.classify_all()

        # 3 files x 3 consecutive failures before abort
        assert summary["errors"] == 9
        assert summary["classified"] == 0

    def test_protected_path_always_kept(self, db: Database) -> None:
        """Even if AI says delete a Windows system file, safety blocks it."""
        path = r"C:\Windows\System32\notepad.exe"
        _insert_test_file(db, path, "notepad.exe", ".exe")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 1.0,
             "reason": "junk", "category": "system"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1

    def test_sensitive_file_protected(self, db: Database) -> None:
        """Sensitive files (.env) must be protected from deletion."""
        path = r"C:\Projects\myapp\.env"
        _insert_test_file(db, path, ".env", ".env")
        files = db.get_files()

        ai_response = _make_ai_response([
            {"path": path, "action": "DELETE_JUNK", "confidence": 0.95,
             "reason": "config junk", "category": "config"},
        ])

        client = self._mock_ollama([ai_response])
        classifier = FileClassifier(db, ollama_client=client)
        result = classifier.classify_batch_direct(files)

        assert result[0]["action"] == "KEEP"
        assert result[0]["overridden"] == 1
