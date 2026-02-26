"""
Tests for the network guard module.

Verifies that DriveMindr's privacy promise is enforced:
no outbound connections except localhost:11434.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from drivemindr.network import (
    LOOPBACK_ADDRS,
    check_outbound_connections,
    get_network_interfaces,
    verify_dns_not_leaking,
)


class TestCheckOutboundConnections:

    @patch("drivemindr.network.psutil.net_connections")
    def test_no_connections_is_safe(self, mock_net) -> None:
        mock_net.return_value = []
        result = check_outbound_connections()
        assert result.safe is True
        assert result.suspicious_connections == []

    @patch("drivemindr.network.psutil.net_connections")
    def test_loopback_connections_are_safe(self, mock_net) -> None:
        conn = MagicMock()
        conn.status = "ESTABLISHED"
        conn.raddr = MagicMock()
        conn.raddr.ip = "127.0.0.1"
        conn.raddr.port = 11434
        conn.laddr = MagicMock()
        conn.laddr.port = 54321
        conn.pid = 1234
        mock_net.return_value = [conn]

        result = check_outbound_connections()
        assert result.safe is True

    @patch("drivemindr.network.psutil.Process")
    @patch("drivemindr.network.psutil.net_connections")
    def test_external_python_connection_flagged(self, mock_net, mock_proc) -> None:
        """A Python process connecting to an external IP should be flagged."""
        conn = MagicMock()
        conn.status = "ESTABLISHED"
        conn.raddr = MagicMock()
        conn.raddr.ip = "8.8.8.8"
        conn.raddr.port = 443
        conn.laddr = MagicMock()
        conn.laddr.port = 54321
        conn.pid = 9999
        mock_net.return_value = [conn]

        proc = MagicMock()
        proc.name.return_value = "python"
        mock_proc.return_value = proc

        result = check_outbound_connections()
        assert result.safe is False
        assert len(result.suspicious_connections) == 1
        assert result.suspicious_connections[0]["remote_ip"] == "8.8.8.8"

    @patch("drivemindr.network.psutil.Process")
    @patch("drivemindr.network.psutil.net_connections")
    def test_non_python_external_connection_not_flagged(self, mock_net, mock_proc) -> None:
        """System processes (chrome, etc.) with external connections are OK."""
        conn = MagicMock()
        conn.status = "ESTABLISHED"
        conn.raddr = MagicMock()
        conn.raddr.ip = "142.250.80.46"
        conn.raddr.port = 443
        conn.laddr = MagicMock()
        conn.laddr.port = 54321
        conn.pid = 5555
        mock_net.return_value = [conn]

        proc = MagicMock()
        proc.name.return_value = "chrome"
        mock_proc.return_value = proc

        result = check_outbound_connections()
        assert result.safe is True

    @patch("drivemindr.network.psutil.net_connections")
    def test_non_established_connections_ignored(self, mock_net) -> None:
        conn = MagicMock()
        conn.status = "LISTEN"
        conn.raddr = None
        mock_net.return_value = [conn]

        result = check_outbound_connections()
        assert result.safe is True

    @patch("drivemindr.network.psutil.net_connections")
    def test_access_denied_returns_warning(self, mock_net) -> None:
        import psutil
        mock_net.side_effect = psutil.AccessDenied(pid=1)

        result = check_outbound_connections()
        assert result.safe is True  # defaults to safe when can't check
        assert len(result.warnings) == 1


class TestVerifyDnsNotLeaking:

    @patch("drivemindr.network.socket.socket")
    def test_connection_failure_means_safe(self, mock_socket_cls) -> None:
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1  # connection refused = safe
        mock_socket_cls.return_value = mock_sock

        assert verify_dns_not_leaking() is True

    @patch("drivemindr.network.socket.socket")
    def test_connection_success_means_interception(self, mock_socket_cls) -> None:
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0  # connection succeeded = suspicious
        mock_socket_cls.return_value = mock_sock

        assert verify_dns_not_leaking() is False


class TestGetNetworkInterfaces:

    @patch("drivemindr.network.psutil.net_if_stats")
    @patch("drivemindr.network.psutil.net_if_addrs")
    def test_filters_loopback(self, mock_addrs, mock_stats) -> None:
        import socket

        lo_addr = MagicMock()
        lo_addr.family = socket.AF_INET
        lo_addr.address = "127.0.0.1"
        lo_addr.netmask = "255.0.0.0"

        eth_addr = MagicMock()
        eth_addr.family = socket.AF_INET
        eth_addr.address = "192.168.1.100"
        eth_addr.netmask = "255.255.255.0"

        mock_addrs.return_value = {
            "lo": [lo_addr],
            "eth0": [eth_addr],
        }

        lo_stat = MagicMock()
        lo_stat.isup = True
        eth_stat = MagicMock()
        eth_stat.isup = True

        mock_stats.return_value = {"lo": lo_stat, "eth0": eth_stat}

        interfaces = get_network_interfaces()
        names = [i["name"] for i in interfaces]
        assert "lo" not in names
        assert "eth0" in names


class TestLoopbackAddrs:

    def test_all_loopback_variants_included(self) -> None:
        assert "127.0.0.1" in LOOPBACK_ADDRS
        assert "::1" in LOOPBACK_ADDRS
        assert "0.0.0.0" in LOOPBACK_ADDRS
