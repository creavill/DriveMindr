"""Tests for utility helpers."""

from drivemindr.utils import clamp, format_bytes, format_count


class TestFormatBytes:

    def test_bytes(self) -> None:
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"

    def test_kilobytes(self) -> None:
        assert "KB" in format_bytes(1024)

    def test_megabytes(self) -> None:
        assert "MB" in format_bytes(1024 * 1024)

    def test_gigabytes(self) -> None:
        assert "GB" in format_bytes(1024 ** 3)

    def test_terabytes(self) -> None:
        assert "TB" in format_bytes(1024 ** 4)

    def test_negative(self) -> None:
        result = format_bytes(-1024)
        assert result.startswith("-")
        assert "KB" in result


class TestFormatCount:

    def test_small(self) -> None:
        assert format_count(42) == "42"

    def test_thousands(self) -> None:
        assert format_count(1234567) == "1,234,567"


class TestClamp:

    def test_within_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_below_low(self) -> None:
        assert clamp(-0.1) == 0.0

    def test_above_high(self) -> None:
        assert clamp(1.5) == 1.0

    def test_custom_bounds(self) -> None:
        assert clamp(15, low=0, high=10) == 10
