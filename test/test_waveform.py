import numpy as np
import pytest

from jaxqualin.waveforms import waveform


def _make_synthetic_waveform(
    A=1.0, phi=0.0, omega=0.5, tau=10.0,
    t_start=-50.0, t_end=100.0, dt=0.1
):
    """Create a simple damped sinusoid waveform for testing."""
    time = np.arange(t_start, t_end, dt)
    h = A * np.exp(-np.abs(time) / tau) * np.exp(1j * (omega * time + phi))
    return time, h


class TestWaveformConstructor:
    def test_basic_attributes(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        assert w.fulltime is time
        assert np.array_equal(np.asarray(w.fullh), h)

    def test_postmerger_time_starts_at_zero(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        assert w.time[0] >= 0.0 or np.isclose(w.time[0], 0.0, atol=0.2)

    def test_h_is_complex(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        assert np.iscomplexobj(w.h)

    def test_hr_hi_consistency(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        np.testing.assert_allclose(
            np.asarray(w.h),
            np.asarray(w.hr) + 1j * np.asarray(w.hi),
            atol=1e-12
        )

    def test_lm_default_none(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        assert w.l is None
        assert w.m is None

    def test_lm_set_in_constructor(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, l=2, m=2, remove_num=0)
        assert w.l == 2
        assert w.m == 2

    def test_custom_t_peak(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, t_peak=0.0, remove_num=0)
        assert w.peaktime == 0.0


class TestArgabsmax:
    def test_finds_peak(self):
        time = np.linspace(-50, 100, 2000)
        h = np.exp(-np.abs(time) / 10.0) * np.exp(1j * 0.5 * time)
        w = waveform(time, h, remove_num=0)
        peak_time = time[w.peakindx]
        assert np.abs(peak_time) < 1.0

    def test_with_remove_num(self):
        time = np.linspace(-50, 100, 2000)
        h = np.zeros_like(time, dtype=complex)
        h[10] = 100.0
        h[1000] = 50.0
        w = waveform(time, h, remove_num=500)
        assert w.peakindx >= 500


class TestUpdatePeaktime:
    def test_sets_t_peak(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        w.update_peaktime(5.0)
        assert w.t_peak == 5.0


class TestPostmerger:
    def test_t_start_slicing(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, t_peak=0.0, remove_num=0)
        t_post, hr_post, hi_post = w.postmerger(10.0)
        assert t_post[0] >= 10.0

    def test_t_end_slicing(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, t_peak=0.0, remove_num=0)
        t_post, hr_post, hi_post = w.postmerger(0.0, 50.0)
        assert t_post[-1] <= 50.0

    def test_output_shapes_match(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, t_peak=0.0, remove_num=0)
        t_post, hr_post, hi_post = w.postmerger(0.0, 50.0)
        assert len(t_post) == len(hr_post) == len(hi_post)


class TestSetLm:
    def test_set_values(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, remove_num=0)
        w.set_lm(3, 2)
        assert w.l == 3
        assert w.m == 2

    def test_overwrite_values(self):
        time, h = _make_synthetic_waveform()
        w = waveform(time, h, l=2, m=2, remove_num=0)
        w.set_lm(4, 4)
        assert w.l == 4
        assert w.m == 4
