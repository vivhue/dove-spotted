from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from monitor.schemas import HRMeasurement


@dataclass
class RPPGConfig:
    min_bpm: float = 42.0
    max_bpm: float = 180.0
    min_window_seconds: float = 8.0
    max_window_seconds: float = 20.0


class RPPGEstimator:
    """Simple FFT-based rPPG estimator from a temporal green-channel signal."""

    def __init__(self, config: RPPGConfig | None = None) -> None:
        self.config = config or RPPGConfig()
        self._samples: deque[tuple[float, float]] = deque()

    def update(self, value: float, timestamp: float) -> HRMeasurement:
        self._samples.append((timestamp, float(value)))
        while self._samples and timestamp - self._samples[0][0] > self.config.max_window_seconds:
            self._samples.popleft()
        return self._estimate()

    def _estimate(self) -> HRMeasurement:
        if len(self._samples) < 90:
            return HRMeasurement(bpm=None, confidence=0.0)

        samples = np.array(self._samples, dtype=np.float64)
        t = samples[:, 0]
        x = samples[:, 1]

        duration = t[-1] - t[0]
        if duration < self.config.min_window_seconds:
            return HRMeasurement(bpm=None, confidence=0.0)

        # Resample to a uniform timeline before FFT.
        observed_fs = len(t) / duration
        target_fs = float(np.clip(observed_fs, 20.0, 60.0))
        count = max(int(duration * target_fs), 128)
        uniform_t = np.linspace(t[0], t[-1], count)
        uniform_x = np.interp(uniform_t, t, x)

        if float(np.std(uniform_x)) < 1e-6:
            return HRMeasurement(bpm=None, confidence=0.0)

        trend = np.linspace(uniform_x[0], uniform_x[-1], uniform_x.size)
        signal = uniform_x - trend
        signal = signal - np.mean(signal)

        window = np.hanning(signal.size)
        spectrum = np.abs(np.fft.rfft(signal * window)) ** 2
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / target_fs)

        min_hz = self.config.min_bpm / 60.0
        max_hz = self.config.max_bpm / 60.0
        band = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(band):
            return HRMeasurement(bpm=None, confidence=0.0)

        band_freqs = freqs[band]
        band_power = spectrum[band]
        peak_index = int(np.argmax(band_power))

        peak_freq = float(band_freqs[peak_index])
        peak_power = float(band_power[peak_index])
        total_power = float(np.sum(band_power)) + 1e-9
        confidence = float(np.clip(peak_power / total_power, 0.0, 1.0))

        bpm = peak_freq * 60.0
        return HRMeasurement(bpm=round(bpm, 1), confidence=round(confidence, 3))

