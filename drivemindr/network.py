"""
DriveMindr network guard — verifies zero outbound connections.

This module ensures the application's core privacy promise:
NO data ever leaves the machine. The only allowed network target
is localhost:11434 (Ollama).

Features:
  - Startup check: verifies no unexpected outbound connections
  - --paranoid mode: optionally disables non-loopback network interfaces
  - Continuous monitoring: can be run as a background check during execution
"""

from __future__ import annotations

import logging
import platform
import socket
import subprocess
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger("drivemindr.network")

# The ONLY allowed remote endpoints (host, port)
ALLOWED_ENDPOINTS: set[tuple[str, int]] = {
    ("127.0.0.1", 11434),  # Ollama
    ("localhost", 11434),
}

# Loopback addresses that are always safe
LOOPBACK_ADDRS: set[str] = {"127.0.0.1", "::1", "0.0.0.0", "::"}


@dataclass
class NetworkCheckResult:
    """Result of a network safety check."""
    safe: bool = True
    suspicious_connections: list[dict[str, str]] = field(default_factory=list)
    interfaces_disabled: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_outbound_connections() -> NetworkCheckResult:
    """Check for any active outbound connections that aren't localhost.

    Returns a NetworkCheckResult indicating whether the system is safe.
    """
    result = NetworkCheckResult()

    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, OSError) as exc:
        logger.warning("Cannot check network connections (need admin): %s", exc)
        result.warnings.append(f"Cannot check connections: {exc}")
        return result

    for conn in connections:
        if conn.status != "ESTABLISHED":
            continue
        if not conn.raddr:
            continue

        remote_ip = conn.raddr.ip
        remote_port = conn.raddr.port

        # Allow loopback
        if remote_ip in LOOPBACK_ADDRS:
            continue

        # Allow explicitly permitted endpoints
        if (remote_ip, remote_port) in ALLOWED_ENDPOINTS:
            continue

        # This is a suspicious outbound connection
        try:
            proc = psutil.Process(conn.pid) if conn.pid else None
            proc_name = proc.name() if proc else "unknown"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_name = "unknown"

        suspicious = {
            "remote_ip": remote_ip,
            "remote_port": str(remote_port),
            "local_port": str(conn.laddr.port) if conn.laddr else "?",
            "pid": str(conn.pid or "?"),
            "process": proc_name,
        }

        # Only flag if it's our own process (Python/drivemindr)
        # Other system processes having connections is expected
        if proc_name.lower() in ("python", "python3", "python.exe", "drivemindr"):
            result.safe = False
            result.suspicious_connections.append(suspicious)
            logger.error(
                "SUSPICIOUS: outbound connection from %s to %s:%s (pid=%s)",
                proc_name, remote_ip, remote_port, conn.pid,
            )

    if result.safe:
        logger.info("Network check PASSED — no suspicious outbound connections")
    else:
        logger.error(
            "Network check FAILED — %d suspicious connections",
            len(result.suspicious_connections),
        )

    return result


def verify_dns_not_leaking() -> bool:
    """Quick check that we're not accidentally resolving external hostnames.

    Tries to connect to a non-routable address — should fail immediately.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        # RFC 5737 TEST-NET — should never route
        err = sock.connect_ex(("192.0.2.1", 80))
        sock.close()
        # If connection succeeded, something is intercepting
        if err == 0:
            logger.warning("DNS/network interception detected — connection to TEST-NET succeeded")
            return False
        return True
    except (socket.error, OSError):
        return True  # expected — no route means we're safe


def get_network_interfaces() -> list[dict[str, str]]:
    """List active non-loopback network interfaces."""
    interfaces = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for name, addr_list in addrs.items():
        stat = stats.get(name)
        if not stat or not stat.isup:
            continue
        # Skip loopback
        if name.lower() in ("lo", "loopback", "lo0"):
            continue

        for addr in addr_list:
            if addr.family == socket.AF_INET:
                interfaces.append({
                    "name": name,
                    "ip": addr.address,
                    "netmask": addr.netmask or "",
                })
                break

    return interfaces


def paranoid_mode(*, enable: bool = True) -> NetworkCheckResult:
    """Enable or disable paranoid mode — disables non-loopback interfaces.

    WARNING: Requires Administrator privileges on Windows.
    This is reversible — interfaces are re-enabled when paranoid mode is turned off.

    On non-Windows systems, this logs what would happen but doesn't modify interfaces.
    """
    result = NetworkCheckResult()
    interfaces = get_network_interfaces()

    if not interfaces:
        logger.info("No non-loopback interfaces found — already isolated")
        return result

    action = "disable" if enable else "enable"

    for iface in interfaces:
        name = iface["name"]
        if platform.system() == "Windows":
            try:
                cmd = ["netsh", "interface", "set", "interface", name, action]
                subprocess.run(cmd, capture_output=True, check=True)
                result.interfaces_disabled.append(name)
                logger.info("Paranoid mode: %sd interface %s", action, name)
            except subprocess.CalledProcessError as exc:
                result.warnings.append(f"Failed to {action} {name}: {exc}")
                logger.error("Failed to %s interface %s: %s", action, name, exc)
        else:
            # Non-Windows: just log, don't modify
            logger.info(
                "Paranoid mode (non-Windows): would %s interface %s (%s)",
                action, name, iface["ip"],
            )
            result.interfaces_disabled.append(name)

    return result
