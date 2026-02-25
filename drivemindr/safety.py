"""
DriveMindr safety engine — the most important module.

Implements four layers of protection that override AI decisions:

  Layer 1: Hardcoded protected paths (AI cannot override)
  Layer 2: Document Guardian (docs/photos/code never auto-deleted)
  Layer 3: Confidence thresholds (low confidence → manual review)
  Layer 4: Pre-execution dry run (nothing runs without user approval)

No network calls. No exceptions. Safety always wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Any

from drivemindr.config import (
    CONFIDENCE_AUTO_APPROVE,
    CONFIDENCE_DELETE_MIN,
    CONFIDENCE_UNCERTAIN,
    GUARDIAN_EXTENSIONS,
    PROTECTED_OWNERS,
    PROTECTED_PATHS,
    SENSITIVE_FILE_PATTERNS,
)

logger = logging.getLogger("drivemindr.safety")


@dataclass
class SafetyVerdict:
    """Result of running a file through the safety engine."""

    original_action: str
    final_action: str
    confidence: float
    is_protected: bool = False
    is_guardian_protected: bool = False
    is_sensitive: bool = False
    overridden: bool = False
    override_reason: str = ""
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return self.original_action != self.final_action


class SafetyEngine:
    """Multi-layer safety engine that has final say over the AI classifier.

    Usage::

        engine = SafetyEngine()
        verdict = engine.check(file_path, ai_action, ai_confidence, owner=...)
        # verdict.final_action is the safe action to use
    """

    def __init__(self) -> None:
        # Pre-compute normalized protected paths for fast lookups
        self._protected: list[PureWindowsPath] = [
            PureWindowsPath(p) for p in PROTECTED_PATHS
        ]
        self._protected_owners_lower: set[str] = {
            o.lower() for o in PROTECTED_OWNERS
        }
        self._sensitive_patterns_lower: list[str] = [
            p.lower() for p in SENSITIVE_FILE_PATTERNS
        ]
        logger.debug(
            "SafetyEngine initialized — %d protected paths, %d protected owners",
            len(self._protected),
            len(self._protected_owners_lower),
        )

    # -- public API -----------------------------------------------------------

    def check(
        self,
        file_path: str,
        ai_action: str,
        ai_confidence: float,
        *,
        owner: str | None = None,
        extension: str | None = None,
    ) -> SafetyVerdict:
        """Run a file through all safety layers and return the verdict.

        Args:
            file_path: Full Windows path of the file.
            ai_action: The action the AI classifier chose.
            ai_confidence: The AI's confidence score (0.0–1.0).
            owner: File owner string (e.g. ``SYSTEM``).
            extension: File extension including dot (e.g. ``.docx``).

        Returns:
            A ``SafetyVerdict`` with the final safe action.
        """
        verdict = SafetyVerdict(
            original_action=ai_action,
            final_action=ai_action,
            confidence=ai_confidence,
        )

        logger.debug(
            "Safety check — path=%s action=%s conf=%.2f owner=%s ext=%s",
            file_path, ai_action, ai_confidence, owner, extension,
        )

        # Layer 1: Hardcoded protected paths
        self._check_protected_path(file_path, verdict)
        if verdict.is_protected:
            return verdict  # short-circuit — nothing else matters

        # Layer 1b: Protected owners
        self._check_protected_owner(owner, verdict)
        if verdict.is_protected:
            return verdict

        # Layer 2: Document Guardian
        self._check_guardian(file_path, extension, verdict)

        # Layer 2b: Sensitive file detection
        self._check_sensitive(file_path, verdict)

        # Layer 3: Confidence thresholds
        self._check_confidence(verdict)

        if verdict.was_modified:
            logger.info(
                "Safety override — path=%s original=%s final=%s reason=%s",
                file_path,
                verdict.original_action,
                verdict.final_action,
                verdict.override_reason,
            )

        return verdict

    def is_path_protected(self, file_path: str) -> bool:
        """Quick check: is this path under a hardcoded protected directory?"""
        normalized = PureWindowsPath(file_path)
        for protected in self._protected:
            try:
                normalized.relative_to(protected)
                return True
            except ValueError:
                continue
        return False

    def is_delete_action(self, action: str) -> bool:
        """Check if an action is a delete variant."""
        return action.upper() in ("DELETE_JUNK", "DELETE_UNUSED", "DELETE")

    def is_sensitive_file(self, file_path: str) -> bool:
        """Check if a file matches sensitive file patterns."""
        lower_path = file_path.lower()
        name = PureWindowsPath(lower_path).name
        for pattern in self._sensitive_patterns_lower:
            if pattern in name:
                return True
        return False

    # -- layer implementations ------------------------------------------------

    def _check_protected_path(self, file_path: str, verdict: SafetyVerdict) -> None:
        """Layer 1: Hardcoded protected paths — AI cannot override."""
        if self.is_path_protected(file_path):
            verdict.is_protected = True
            verdict.final_action = "KEEP"
            verdict.overridden = True
            verdict.override_reason = "Hardcoded protected path — cannot be modified"
            verdict.warnings.append(f"PROTECTED PATH: {file_path}")
            logger.warning("Layer 1 block — protected path: %s", file_path)

    def _check_protected_owner(self, owner: str | None, verdict: SafetyVerdict) -> None:
        """Layer 1b: Protected owner — SYSTEM/TrustedInstaller files are untouchable."""
        if owner and owner.lower() in self._protected_owners_lower:
            verdict.is_protected = True
            verdict.final_action = "KEEP"
            verdict.overridden = True
            verdict.override_reason = f"Protected owner: {owner}"
            verdict.warnings.append(f"PROTECTED OWNER: {owner}")
            logger.warning("Layer 1b block — protected owner: %s", owner)

    def _check_guardian(self, file_path: str, extension: str | None, verdict: SafetyVerdict) -> None:
        """Layer 2: Document Guardian — docs/photos/code can never be auto-deleted."""
        if not self.is_delete_action(verdict.final_action):
            return  # only intervene on delete actions

        ext = extension or PureWindowsPath(file_path).suffix
        if ext and ext.lower() in GUARDIAN_EXTENSIONS:
            verdict.is_guardian_protected = True
            verdict.final_action = "KEEP"
            verdict.overridden = True
            verdict.override_reason = (
                f"Document Guardian — {ext} files cannot be deleted, only moved/archived"
            )
            verdict.needs_review = True
            verdict.warnings.append(f"GUARDIAN: {ext} file protected from deletion")
            logger.info(
                "Layer 2 override — guardian protected: %s (ext=%s)", file_path, ext
            )

    def _check_sensitive(self, file_path: str, verdict: SafetyVerdict) -> None:
        """Layer 2b: Sensitive file detection — flag for maximum protection."""
        if self.is_sensitive_file(file_path):
            verdict.is_sensitive = True
            verdict.warnings.append(f"SENSITIVE: {file_path} matches sensitive pattern")
            if self.is_delete_action(verdict.final_action):
                verdict.final_action = "KEEP"
                verdict.overridden = True
                verdict.override_reason = "Sensitive file — cannot be deleted"
                verdict.needs_review = True
            logger.info("Layer 2b — sensitive file flagged: %s", file_path)

    def _check_confidence(self, verdict: SafetyVerdict) -> None:
        """Layer 3: Confidence thresholds — low confidence routes to manual review."""
        # Delete actions need extra-high confidence
        if self.is_delete_action(verdict.final_action):
            if verdict.confidence < CONFIDENCE_DELETE_MIN:
                verdict.final_action = "KEEP"
                verdict.overridden = True
                verdict.override_reason = (
                    f"Delete requires confidence >= {CONFIDENCE_DELETE_MIN}, "
                    f"got {verdict.confidence:.2f}"
                )
                verdict.needs_review = True
                logger.info(
                    "Layer 3 override — delete confidence too low: %.2f < %.2f",
                    verdict.confidence,
                    CONFIDENCE_DELETE_MIN,
                )
                return

        # General low-confidence → manual review
        if verdict.confidence < CONFIDENCE_UNCERTAIN:
            verdict.needs_review = True
            verdict.warnings.append(
                f"UNCERTAIN: confidence {verdict.confidence:.2f} < {CONFIDENCE_UNCERTAIN}"
            )
            logger.info(
                "Layer 3 flag — uncertain confidence: %.2f", verdict.confidence
            )
        elif verdict.confidence < CONFIDENCE_AUTO_APPROVE:
            verdict.needs_review = True
            logger.debug(
                "Layer 3 flag — needs review: confidence %.2f < %.2f",
                verdict.confidence,
                CONFIDENCE_AUTO_APPROVE,
            )
