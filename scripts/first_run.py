"""
DriveMindr first-run setup wizard.

Checks that the system is ready to run DriveMindr:
  1. Python version >= 3.11
  2. Ollama installed and running
  3. Required model pulled
  4. Available drives detected
  5. Administrator privileges (for symlink operations)
  6. Network isolation verified

Can be run standalone: python scripts/first_run.py
Or via CLI: drivemindr setup
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Ensure drivemindr is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from drivemindr.classifier import OllamaClient
from drivemindr.config import OLLAMA_HOST, OLLAMA_MODEL
from drivemindr.network import check_outbound_connections, get_network_interfaces
from drivemindr.symlinks import is_admin


def _print_check(label: str, passed: bool, detail: str = "") -> None:
    """Print a check result with a pass/fail indicator."""
    icon = "[OK]" if passed else "[!!]"
    msg = f"  {icon} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def check_python() -> bool:
    """Verify Python version."""
    version = sys.version_info
    ok = version >= (3, 11)
    _print_check(
        "Python version",
        ok,
        f"{version.major}.{version.minor}.{version.micro}"
        + ("" if ok else " (need 3.11+)"),
    )
    return ok


def check_platform() -> bool:
    """Report platform info."""
    system = platform.system()
    is_windows = system == "Windows"
    _print_check(
        "Platform",
        True,
        f"{system} {platform.release()}"
        + ("" if is_windows else " (Windows recommended for full functionality)"),
    )
    return True


def check_ollama() -> tuple[bool, bool]:
    """Check if Ollama is installed and the model is available."""
    client = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL)

    ollama_up = client.is_available()
    _print_check(
        "Ollama running",
        ollama_up,
        f"at {OLLAMA_HOST}" if ollama_up else f"not found at {OLLAMA_HOST} — run 'ollama serve'",
    )

    model_ready = False
    if ollama_up:
        model_ready = client.has_model()
        _print_check(
            f"Model '{OLLAMA_MODEL}'",
            model_ready,
            "available" if model_ready else f"not found — run 'ollama pull {OLLAMA_MODEL}'",
        )
    else:
        _print_check(
            f"Model '{OLLAMA_MODEL}'",
            False,
            "skipped (Ollama not running)",
        )

    return ollama_up, model_ready


def check_drives() -> dict[str, bool]:
    """Check for available drives (Windows-specific)."""
    drives: dict[str, bool] = {}

    if platform.system() == "Windows":
        for letter in ["C", "D", "E", "F"]:
            path = Path(f"{letter}:\\")
            exists = path.exists()
            if exists:
                try:
                    import shutil
                    usage = shutil.disk_usage(str(path))
                    free_gb = usage.free / (1024 ** 3)
                    total_gb = usage.total / (1024 ** 3)
                    detail = f"{free_gb:.1f} GB free / {total_gb:.1f} GB total"
                except OSError:
                    detail = "accessible"
                drives[letter] = True
                _print_check(f"Drive {letter}:", True, detail)
            elif letter in ("C", "D"):
                drives[letter] = False
                _print_check(
                    f"Drive {letter}:",
                    False,
                    "not found" + (" (required for file migration)" if letter == "D" else ""),
                )
    else:
        _print_check("Drive detection", True, "non-Windows — using current filesystem")
        drives["C"] = True

    return drives


def check_admin() -> bool:
    """Check for Administrator privileges."""
    admin = is_admin()
    _print_check(
        "Administrator privileges",
        admin,
        "granted" if admin else "not elevated — symlink operations will require admin",
    )
    return admin


def check_network() -> bool:
    """Verify no suspicious outbound connections."""
    result = check_outbound_connections()
    _print_check(
        "Network isolation",
        result.safe,
        "no suspicious outbound connections"
        if result.safe
        else f"{len(result.suspicious_connections)} suspicious connections detected",
    )

    interfaces = get_network_interfaces()
    if interfaces:
        names = ", ".join(i["name"] for i in interfaces)
        _print_check(
            "Active interfaces",
            True,
            f"{len(interfaces)} active ({names}) — use --paranoid to disable",
        )

    return result.safe


def check_dependencies() -> bool:
    """Check that required Python packages are installed."""
    required = ["typer", "rich", "psutil"]
    optional = ["streamlit"]
    all_ok = True

    for pkg in required:
        try:
            __import__(pkg)
            _print_check(f"Package '{pkg}'", True, "installed")
        except ImportError:
            _print_check(f"Package '{pkg}'", False, "missing — pip install drivemindr")
            all_ok = False

    for pkg in optional:
        try:
            __import__(pkg)
            _print_check(f"Package '{pkg}' (optional)", True, "installed")
        except ImportError:
            _print_check(
                f"Package '{pkg}' (optional)",
                True,  # optional, so still OK
                "not installed — pip install drivemindr[dashboard]",
            )

    return all_ok


def run_wizard() -> bool:
    """Run the complete first-run setup wizard.

    Returns True if all critical checks pass.
    """
    print("\n" + "=" * 60)
    print("  DriveMindr — First Run Setup Wizard")
    print("  Privacy-First | Fully Offline | Local AI Only")
    print("=" * 60 + "\n")

    all_ok = True

    print("System Checks:")
    print("-" * 40)
    all_ok &= check_python()
    check_platform()
    all_ok &= check_dependencies()

    print("\nOllama (Local AI):")
    print("-" * 40)
    ollama_up, model_ready = check_ollama()
    # Ollama not being ready is a warning, not a blocker for scan

    print("\nStorage:")
    print("-" * 40)
    drives = check_drives()

    print("\nPrivileges:")
    print("-" * 40)
    admin = check_admin()

    print("\nPrivacy & Network:")
    print("-" * 40)
    net_ok = check_network()

    print("\n" + "=" * 60)
    if all_ok:
        print("  All critical checks passed!")
        print()
        print("  Quick start:")
        print("    1. drivemindr scan C:\\")
        print("    2. drivemindr classify")
        print("    3. drivemindr dashboard")
        print("    4. drivemindr execute --dry-run")
        if not ollama_up:
            print()
            print("  Note: Start Ollama before running 'classify':")
            print("    ollama serve")
            print(f"    ollama pull {OLLAMA_MODEL}")
        if not admin:
            print()
            print("  Note: Run as Administrator for app migration (symlinks).")
    else:
        print("  Some checks failed — see above for details.")
    print("=" * 60 + "\n")

    return all_ok


if __name__ == "__main__":
    success = run_wizard()
    sys.exit(0 if success else 1)
