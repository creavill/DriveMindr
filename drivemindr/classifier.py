"""
DriveMindr AI classifier — local Ollama integration for file classification.

Connects ONLY to localhost:11434. Never sends file contents. Never makes
external network calls. Uses stdlib urllib so no pip install needed for the
Ollama client — keeping the "no internet required" promise.

Flow:
  1. Pull unclassified files from SQLite in batches
  2. Build metadata-only prompt for each batch
  3. Send to local Ollama, parse structured JSON response
  4. Run every classification through SafetyEngine before storing
  5. Store final (possibly overridden) classification in DB
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from drivemindr.config import (
    OLLAMA_BATCH_SIZE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from drivemindr.database import Database
from drivemindr.safety import SafetyEngine

logger = logging.getLogger("drivemindr.classifier")

# ---------------------------------------------------------------------------
# Ollama system prompt — the AI only ever sees metadata, never contents
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Windows storage management assistant. You analyze file metadata
and classify files into categories. You NEVER see file contents. You only see:
name, extension, size, path, last_accessed, last_modified.

Classify each file as one of: KEEP, MOVE_APP, MOVE_DATA, DELETE_JUNK,
DELETE_UNUSED, ARCHIVE. Include a confidence score (0.0-1.0) and a brief reason.

RULES:
- Documents (.doc, .pdf, .txt, etc.) are NEVER classified as DELETE
- Photos and videos are NEVER classified as DELETE
- Source code is NEVER classified as DELETE
- Installer packages (.msi, .exe in Downloads) CAN be DELETE_JUNK
- Temp files, caches, logs older than 30 days CAN be DELETE_JUNK
- Apps not accessed in 6+ months CAN be DELETE_UNUSED
- When uncertain, prefer KEEP over DELETE

Respond ONLY with a JSON array — no markdown fences, no extra text:
[{"path": "...", "action": "...", "confidence": 0.0, "reason": "...", "category": "..."}]
"""

# Valid actions the AI may return
VALID_ACTIONS = frozenset({
    "KEEP", "MOVE_APP", "MOVE_DATA", "DELETE_JUNK", "DELETE_UNUSED", "ARCHIVE",
})


@dataclass
class ClassificationResult:
    """A single file classification from the AI (before safety override)."""
    path: str
    action: str
    confidence: float
    reason: str
    category: str


class OllamaClient:
    """Minimal Ollama REST client using only stdlib. Localhost only."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self._generate_url = f"{self.host}/api/generate"
        logger.debug("OllamaClient configured — host=%s model=%s", self.host, self.model)

    def is_available(self) -> bool:
        """Check if Ollama is running on localhost."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                available = resp.status == 200
                logger.info("Ollama availability check: %s", "UP" if available else "DOWN")
                return available
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Ollama not available at %s: %s", self.host, exc)
            return False

    def has_model(self, model: str | None = None) -> bool:
        """Check if the required model is pulled."""
        target = model or self.model
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match exact name or name without :latest tag
                found = any(
                    m == target or m.split(":")[0] == target.split(":")[0]
                    for m in models
                )
                logger.info(
                    "Model check — target=%s found=%s available=%s",
                    target, found, models,
                )
                return found
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not check models: %s", exc)
            return False

    def generate(self, prompt: str, *, system: str = SYSTEM_PROMPT) -> str:
        """Send a prompt to Ollama and return the full response text.

        Uses the /api/generate endpoint with stream=false for simplicity.
        """
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.1,  # low temp for consistent structured output
                "num_predict": 4096,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            self._generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        logger.debug("Sending prompt to Ollama (%d bytes)", len(payload))
        start = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
                elapsed = time.perf_counter() - start
                response_text = body.get("response", "")
                logger.info(
                    "Ollama responded in %.1fs (%d chars)",
                    elapsed, len(response_text),
                )
                logger.debug("Ollama raw response: %s", response_text[:500])
                return response_text
        except urllib.error.URLError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise ConnectionError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            logger.error("Ollama returned invalid JSON: %s", exc)
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc


def _build_batch_prompt(files: list[sqlite3.Row]) -> str:
    """Build a metadata-only prompt for a batch of files.

    NEVER includes file contents — only name, extension, size, path, dates.
    """
    lines = ["Classify these files:\n"]
    for f in files:
        lines.append(
            f"- path: {f['path']}, name: {f['name']}, ext: {f['extension']}, "
            f"size: {f['size_bytes']} bytes, modified: {f['modified']}, "
            f"accessed: {f['accessed']}"
        )
    return "\n".join(lines)


def _parse_response(text: str, expected_count: int) -> list[ClassificationResult]:
    """Parse the AI's JSON response into ClassificationResult objects.

    Tolerant of common LLM quirks (markdown fences, trailing commas, etc.).
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Attempt to extract just the JSON array if there's surrounding text
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Remove trailing commas before ] (common LLM mistake)
    cleaned = re.sub(r",\s*\]", "]", cleaned)

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse AI response as JSON: %s\nRaw: %s", exc, text[:500])
        return []

    if not isinstance(raw, list):
        logger.error("AI response is not a JSON array: %s", type(raw))
        return []

    results: list[ClassificationResult] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item in AI response: %s", item)
            continue

        action = str(item.get("action", "KEEP")).upper()
        if action not in VALID_ACTIONS:
            logger.warning("Invalid action '%s' from AI — defaulting to KEEP", action)
            action = "KEEP"

        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            logger.warning("Invalid confidence '%s' — defaulting to 0.0", confidence)
            confidence = 0.0

        results.append(ClassificationResult(
            path=str(item.get("path", "")),
            action=action,
            confidence=confidence,
            reason=str(item.get("reason", "")),
            category=str(item.get("category", "")),
        ))

    if len(results) != expected_count:
        logger.warning(
            "AI returned %d classifications but expected %d",
            len(results), expected_count,
        )

    return results


class FileClassifier:
    """Orchestrates AI classification with safety overrides.

    Usage::

        classifier = FileClassifier(db)
        if not classifier.preflight_check():
            print("Ollama not available")
            return
        stats = classifier.classify_all()
    """

    def __init__(
        self,
        db: Database,
        *,
        ollama_client: OllamaClient | None = None,
        safety_engine: SafetyEngine | None = None,
        batch_size: int = OLLAMA_BATCH_SIZE,
    ) -> None:
        self.db = db
        self.ollama = ollama_client or OllamaClient()
        self.safety = safety_engine or SafetyEngine()
        self.batch_size = batch_size
        self._classified = 0
        self._overridden = 0
        self._errors = 0
        self._batches = 0

    def preflight_check(self) -> dict[str, bool]:
        """Verify Ollama is running and the model is available.

        Returns a dict with ``ollama_up`` and ``model_ready`` booleans.
        """
        result = {"ollama_up": False, "model_ready": False}
        result["ollama_up"] = self.ollama.is_available()
        if result["ollama_up"]:
            result["model_ready"] = self.ollama.has_model()
        logger.info("Preflight check: %s", result)
        return result

    def classify_all(self, *, progress_callback: Any = None) -> dict[str, int]:
        """Classify all unclassified files in the database.

        Args:
            progress_callback: Optional callable(classified, overridden, errors)

        Returns:
            Summary dict with ``classified``, ``overridden``, ``errors``, ``batches``.
        """
        logger.info("Starting classification run — batch_size=%d", self.batch_size)

        consecutive_failures = 0
        max_consecutive_failures = 3

        while True:
            # Fetch next batch — always offset=0 because classified files
            # are excluded by the LEFT JOIN, so the result set shrinks.
            files = self._get_unclassified_batch()
            if not files:
                break

            self._batches += 1
            classified_before = self._classified
            logger.info(
                "Processing batch %d — %d files",
                self._batches, len(files),
            )

            self._classify_batch(files)

            # Track consecutive failures to avoid infinite loops when
            # Ollama is down — unclassified files would reappear forever.
            if self._classified == classified_before:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Aborting — %d consecutive batch failures",
                        consecutive_failures,
                    )
                    break
            else:
                consecutive_failures = 0

            if progress_callback:
                progress_callback(self._classified, self._overridden, self._errors)

        summary = {
            "classified": self._classified,
            "overridden": self._overridden,
            "errors": self._errors,
            "batches": self._batches,
        }
        logger.info("Classification complete: %s", summary)
        return summary

    def classify_batch_direct(
        self, files: list[sqlite3.Row], ai_results: list[ClassificationResult] | None = None,
    ) -> list[dict[str, Any]]:
        """Classify a specific batch of files. Optionally pass pre-computed AI results.

        Used for testing and for callers that want fine-grained control.
        Returns a list of classification dicts (with safety verdicts applied).
        """
        if ai_results is None:
            prompt = _build_batch_prompt(files)
            response = self.ollama.generate(prompt)
            ai_results = _parse_response(response, len(files))

        return self._apply_safety_and_store(files, ai_results)

    # -- internals ------------------------------------------------------------

    def _get_unclassified_batch(self) -> list[sqlite3.Row]:
        """Fetch files that don't yet have a classification."""
        sql = """
        SELECT f.* FROM files f
        LEFT JOIN classifications c ON f.id = c.file_id
        WHERE c.id IS NULL AND f.is_dir = 0
        ORDER BY f.id
        LIMIT ?
        """
        return self.db.conn.execute(sql, (self.batch_size,)).fetchall()

    def _classify_batch(self, files: list[sqlite3.Row]) -> None:
        """Send a batch to Ollama and store safety-checked results."""
        prompt = _build_batch_prompt(files)

        try:
            response = self.ollama.generate(prompt)
            ai_results = _parse_response(response, len(files))
        except (ConnectionError, ValueError) as exc:
            logger.error("Batch %d failed: %s — marking as errors", self._batches, exc)
            self._errors += len(files)
            return

        self._apply_safety_and_store(files, ai_results)

    def _apply_safety_and_store(
        self, files: list[sqlite3.Row], ai_results: list[ClassificationResult],
    ) -> list[dict[str, Any]]:
        """Run safety engine on AI results and store classifications in DB."""
        # Build a lookup from path → AI result
        result_map: dict[str, ClassificationResult] = {}
        for r in ai_results:
            result_map[r.path] = r

        stored: list[dict[str, Any]] = []

        for f in files:
            path = f["path"]
            ai = result_map.get(path)

            if ai is None:
                # AI didn't return a result for this file — default to KEEP
                logger.warning("No AI result for %s — defaulting to KEEP", path)
                ai = ClassificationResult(
                    path=path, action="KEEP", confidence=0.0,
                    reason="No AI classification returned", category="unknown",
                )
                self._errors += 1

            # Run through safety engine — this is where overrides happen
            verdict = self.safety.check(
                file_path=path,
                ai_action=ai.action,
                ai_confidence=ai.confidence,
                owner=f["owner"],
                extension=f["extension"],
            )

            if verdict.overridden:
                self._overridden += 1

            classification = {
                "file_id": f["id"],
                "action": verdict.final_action,
                "confidence": ai.confidence,
                "reason": ai.reason,
                "category": ai.category,
                "overridden": 1 if verdict.overridden else 0,
                "override_reason": verdict.override_reason or None,
            }

            try:
                self._store_classification(classification)
                self._classified += 1
                stored.append(classification)
            except Exception:
                logger.exception("Failed to store classification for %s", path)
                self._errors += 1

        return stored

    def _store_classification(self, data: dict[str, Any]) -> None:
        """Insert or update a classification record."""
        sql = """
        INSERT INTO classifications
            (file_id, action, confidence, reason, category, overridden, override_reason)
        VALUES
            (:file_id, :action, :confidence, :reason, :category, :overridden, :override_reason)
        ON CONFLICT(file_id) DO UPDATE SET
            action=excluded.action, confidence=excluded.confidence,
            reason=excluded.reason, category=excluded.category,
            overridden=excluded.overridden, override_reason=excluded.override_reason,
            classified_at=datetime('now','localtime')
        """
        with self.db.transaction() as cur:
            cur.execute(sql, data)
