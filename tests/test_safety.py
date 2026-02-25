"""
Critical safety tests — these MUST pass before any real execution.

Tests cover all four safety layers:
  Layer 1: Hardcoded protected paths
  Layer 2: Document Guardian
  Layer 3: Confidence thresholds
  Sensitive file detection
"""

import pytest

from drivemindr.safety import SafetyEngine, SafetyVerdict


@pytest.fixture
def engine() -> SafetyEngine:
    return SafetyEngine()


# ---------------------------------------------------------------------------
# Layer 1: Hardcoded protected paths
# ---------------------------------------------------------------------------

class TestProtectedPaths:

    def test_cannot_delete_windows_directory(self, engine: SafetyEngine) -> None:
        """C:\\Windows\\System32\\notepad.exe must ALWAYS be blocked."""
        verdict = engine.check(
            r"C:\Windows\System32\notepad.exe",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True
        assert verdict.overridden is True

    def test_cannot_move_windows_directory(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Windows\explorer.exe",
            ai_action="MOVE_DATA",
            ai_confidence=0.99,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_cannot_touch_boot_files(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Boot\BCD",
            ai_action="DELETE_JUNK",
            ai_confidence=1.0,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_cannot_touch_program_data_microsoft(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup",
            ai_action="DELETE_UNUSED",
            ai_confidence=0.95,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_cannot_touch_recovery_partition(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Recovery\WindowsRE\winre.wim",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_unprotected_path_passes_through(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\Downloads\old-installer.msi",
            ai_action="DELETE_JUNK",
            ai_confidence=0.95,
        )
        # Not protected, high confidence — stays as-is
        assert verdict.is_protected is False


# ---------------------------------------------------------------------------
# Layer 1b: Protected owners
# ---------------------------------------------------------------------------

class TestProtectedOwners:

    def test_system_owner_blocks_delete(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\SomeFile.dll",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            owner="SYSTEM",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_trusted_installer_blocks_delete(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\SomeFile.dll",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            owner="NT SERVICE\\TrustedInstaller",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_regular_owner_allows_action(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\temp.tmp",
            ai_action="DELETE_JUNK",
            ai_confidence=0.95,
            owner="Conner",
        )
        assert verdict.is_protected is False


# ---------------------------------------------------------------------------
# Layer 2: Document Guardian
# ---------------------------------------------------------------------------

class TestDocumentGuardian:

    def test_cannot_delete_docx(self, engine: SafetyEngine) -> None:
        """AI classifies a .docx as DELETE_JUNK → safety must override to KEEP."""
        verdict = engine.check(
            r"C:\Users\Conner\Documents\report.docx",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            extension=".docx",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True
        assert verdict.overridden is True

    def test_cannot_delete_pdf(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\tax-return.pdf",
            ai_action="DELETE_UNUSED",
            ai_confidence=0.99,
            extension=".pdf",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True

    def test_cannot_delete_photo(self, engine: SafetyEngine) -> None:
        """AI classifies a .jpg as DELETE_JUNK → safety must override to KEEP."""
        verdict = engine.check(
            r"C:\Users\Conner\Photos\vacation.jpg",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            extension=".jpg",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True

    def test_cannot_delete_video(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\Videos\birthday.mp4",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            extension=".mp4",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True

    def test_cannot_delete_source_code(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Projects\app\main.py",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
            extension=".py",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True

    def test_guardian_allows_move(self, engine: SafetyEngine) -> None:
        """Documents CAN be moved — guardian only blocks deletion."""
        verdict = engine.check(
            r"C:\Users\Conner\Documents\report.docx",
            ai_action="MOVE_DATA",
            ai_confidence=0.9,
            extension=".docx",
        )
        assert verdict.final_action == "MOVE_DATA"
        assert verdict.is_guardian_protected is False

    def test_guardian_allows_archive(self, engine: SafetyEngine) -> None:
        """Documents CAN be archived — guardian only blocks deletion."""
        verdict = engine.check(
            r"C:\Users\Conner\old-project.zip",
            ai_action="ARCHIVE",
            ai_confidence=0.9,
            extension=".zip",
        )
        assert verdict.final_action == "ARCHIVE"

    def test_guardian_infers_extension_from_path(self, engine: SafetyEngine) -> None:
        """If extension is not explicitly passed, it's inferred from the path."""
        verdict = engine.check(
            r"C:\Users\Conner\Documents\notes.txt",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_guardian_protected is True


# ---------------------------------------------------------------------------
# Layer 2b: Sensitive file detection
# ---------------------------------------------------------------------------

class TestSensitiveFiles:

    def test_env_file_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Projects\myapp\.env",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True
        assert verdict.final_action == "KEEP"

    def test_env_local_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Projects\myapp\.env.local",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True
        assert verdict.final_action == "KEEP"

    def test_api_key_file_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\api_key.txt",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True

    def test_private_key_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\.ssh\id_rsa",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True
        assert verdict.final_action == "KEEP"

    def test_credentials_file_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Users\Conner\.aws\credentials",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True

    def test_pem_file_detected(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\certs\server.pem",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.is_sensitive is True

    def test_normal_file_not_sensitive(self, engine: SafetyEngine) -> None:
        assert engine.is_sensitive_file(r"C:\Users\Conner\readme.txt") is False


# ---------------------------------------------------------------------------
# Layer 3: Confidence thresholds
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:

    def test_low_confidence_routes_to_review(self, engine: SafetyEngine) -> None:
        """Confidence 0.5 for a DELETE → must route to MANUAL_REVIEW / KEEP."""
        verdict = engine.check(
            r"C:\Users\Conner\Downloads\something.tmp",
            ai_action="DELETE_JUNK",
            ai_confidence=0.5,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.needs_review is True
        assert verdict.overridden is True

    def test_delete_requires_high_confidence(self, engine: SafetyEngine) -> None:
        """Delete with confidence 0.80 (< 0.85 threshold) → blocked."""
        verdict = engine.check(
            r"C:\Users\Conner\Downloads\old.tmp",
            ai_action="DELETE_JUNK",
            ai_confidence=0.80,
        )
        assert verdict.final_action == "KEEP"
        assert verdict.needs_review is True

    def test_delete_allowed_at_high_confidence(self, engine: SafetyEngine) -> None:
        """Delete with confidence 0.90 (>= 0.85 threshold) → allowed."""
        verdict = engine.check(
            r"C:\Users\Conner\Downloads\old.tmp",
            ai_action="DELETE_JUNK",
            ai_confidence=0.90,
        )
        assert verdict.final_action == "DELETE_JUNK"
        assert verdict.overridden is False

    def test_uncertain_confidence_flagged(self, engine: SafetyEngine) -> None:
        """Confidence < 0.4 → flagged as uncertain."""
        verdict = engine.check(
            r"C:\Users\Conner\misc\unknown-file",
            ai_action="MOVE_DATA",
            ai_confidence=0.3,
        )
        assert verdict.needs_review is True
        assert any("UNCERTAIN" in w for w in verdict.warnings)

    def test_moderate_confidence_needs_review(self, engine: SafetyEngine) -> None:
        """Confidence between 0.4 and 0.7 → needs review but not uncertain."""
        verdict = engine.check(
            r"C:\Users\Conner\misc\file.dat",
            ai_action="MOVE_DATA",
            ai_confidence=0.55,
        )
        assert verdict.needs_review is True

    def test_high_confidence_move_auto_approved(self, engine: SafetyEngine) -> None:
        """Confidence >= 0.7 for a non-delete → no review needed."""
        verdict = engine.check(
            r"C:\Users\Conner\misc\file.dat",
            ai_action="MOVE_DATA",
            ai_confidence=0.85,
        )
        assert verdict.needs_review is False
        assert verdict.overridden is False


# ---------------------------------------------------------------------------
# Composite scenarios
# ---------------------------------------------------------------------------

class TestCompositeScenarios:

    def test_protected_path_overrides_everything(self, engine: SafetyEngine) -> None:
        """Protected path wins even if confidence is perfect."""
        verdict = engine.check(
            r"C:\Windows\System32\driver.sys",
            ai_action="DELETE_JUNK",
            ai_confidence=1.0,
            owner="Conner",
        )
        assert verdict.final_action == "KEEP"
        assert verdict.is_protected is True

    def test_verdict_tracks_original_action(self, engine: SafetyEngine) -> None:
        verdict = engine.check(
            r"C:\Windows\notepad.exe",
            ai_action="DELETE_JUNK",
            ai_confidence=0.99,
        )
        assert verdict.original_action == "DELETE_JUNK"
        assert verdict.final_action == "KEEP"
        assert verdict.was_modified is True
