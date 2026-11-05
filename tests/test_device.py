"""
Tests for device detection.
"""


def test_get_device():
    """Test device detection returns valid device."""
    from protein_entropy.device import get_device

    device = get_device()
    assert device in ["cuda", "mps", "cpu"]


def test_clear_gpu_memory():
    """Test clearing GPU memory doesn't raise exceptions."""
    from protein_entropy.device import clear_gpu_memory

    # Should not raise any exceptions
    clear_gpu_memory()


def test_get_gpu_memory_info():
    """Test GPU memory info returns valid data or None."""
    from protein_entropy.device import get_gpu_memory_info

    info = get_gpu_memory_info()

    if info is not None:
        # If we have GPU info, it should have these keys
        assert "allocated" in info
        assert "reserved" in info
        assert "free" in info
        assert "total" in info

        # Memory values should be non-negative
        assert info["allocated"] >= 0
        assert info["reserved"] >= 0
        assert info["free"] >= 0
        assert info["total"] > 0
