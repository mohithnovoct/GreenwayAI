from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency in some environments.
    pynvml = None

logger = logging.getLogger("greenwayai.telemetry")


@dataclass
class PowerSample:
    timestamp: float
    power_watts: float


class NVMLPowerSampler:
    """Polls local GPU power at 2Hz and attributes power to active requests."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._samples: list[PowerSample] = []
        self._inflight: dict[str, dict[str, float]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._device_handle: Any = None
        self._mock_mode = False
        self.telemetry_source = "unknown"

    def _mock_sample_watts(self) -> float:
        return 75.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        if pynvml is None:
            self._mock_mode = True
            self.telemetry_source = "mock_nvml"
            logger.warning("NVML not installed; running in mock power-sampling mode.")
            self._active = True
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
            return
        try:
            pynvml.nvmlInit()
            self._device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._mock_mode = False
            self.telemetry_source = "real_nvml"
        except Exception as exc:
            self._device_handle = None
            self._mock_mode = True
            self.telemetry_source = "mock_nvml"
            logger.warning("NVML unavailable (%s); running in mock-synthetic mode for CI/dev.", type(exc).__name__)
            self._active = True
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
            return
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._active = False
        if not self._mock_mode and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def register_request(self, request_id: str, *, start_ts: float | None = None, end_ts: float | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._inflight[request_id] = {
                "start_ts": start_ts if start_ts is not None else now,
                "end_ts": end_ts if end_ts is not None else now,
                "completed": 0.0,
                "estimated": 0.0,
                "energy_kwh": 0.0,
            }

    def complete_request(self, request_id: str, *, end_ts: float | None = None) -> None:
        with self._lock:
            entry = self._inflight.get(request_id)
            if entry is None:
                return
            entry["end_ts"] = end_ts if end_ts is not None else time.monotonic()
            entry["completed"] = 1.0
            self._recalculate_energy_locked()

    def add_sample(self, timestamp: float, power_watts: float) -> None:
        with self._lock:
            self._samples.append(PowerSample(timestamp=timestamp, power_watts=float(power_watts)))
            self._prune_samples_locked()

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            if self._mock_mode:
                self.add_sample(time.monotonic(), self._mock_sample_watts())
                continue
            if self._device_handle is None:
                continue
            try:
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self._device_handle) / 1000.0
            except Exception:
                continue
            self.add_sample(time.monotonic(), power_watts)

    def estimate_request_energy_kwh(self, request_id: str) -> float:
        with self._lock:
            request = self._inflight.get(request_id)
            if request is None:
                return 0.0
            self._recalculate_energy_locked()
            request["estimated"] = 1.0
            energy_kwh = request["energy_kwh"]
            self._prune_completed_locked()
            return energy_kwh

    def _request_end_locked(self, request: dict[str, float], now: float) -> float:
        end_ts = request["end_ts"]
        return end_ts if end_ts > request["start_ts"] or request["completed"] else now

    def _recalculate_energy_locked(self) -> None:
        now = time.monotonic()
        requests = list(self._inflight.values())
        for request in requests:
            start_ts = request["start_ts"]
            end_ts = self._request_end_locked(request, now)
            total_joules = 0.0
            for index, sample in enumerate(self._samples):
                sample_start = sample.timestamp
                next_sample_ts = self._samples[index + 1].timestamp if index + 1 < len(self._samples) else end_ts
                overlap_start = max(start_ts, sample_start)
                overlap_end = min(end_ts, next_sample_ts)
                if overlap_end <= overlap_start:
                    continue
                boundaries = {overlap_start, overlap_end}
                for other in requests:
                    other_start = other["start_ts"]
                    other_end = self._request_end_locked(other, now)
                    if overlap_start < other_start < overlap_end:
                        boundaries.add(other_start)
                    if overlap_start < other_end < overlap_end:
                        boundaries.add(other_end)
                ordered_boundaries = sorted(boundaries)
                for boundary_index in range(len(ordered_boundaries) - 1):
                    segment_start = ordered_boundaries[boundary_index]
                    segment_end = ordered_boundaries[boundary_index + 1]
                    midpoint = (segment_start + segment_end) / 2
                    active_count = sum(
                        other["start_ts"] <= midpoint < self._request_end_locked(other, now)
                        for other in requests
                    )
                    if active_count and start_ts <= midpoint < end_ts:
                        total_joules += sample.power_watts * (segment_end - segment_start) / active_count
            request["energy_kwh"] = total_joules / 3_600_000.0

    def _prune_samples_locked(self) -> None:
        if not self._inflight:
            self._samples.clear()
            return
        earliest_start = min(request["start_ts"] for request in self._inflight.values())
        self._samples = [sample for sample in self._samples if sample.timestamp >= earliest_start]

    def _prune_completed_locked(self) -> None:
        if self._inflight and all(request["completed"] and request["estimated"] for request in self._inflight.values()):
            self._inflight.clear()
            self._samples.clear()

    def get_inflight_requests(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {key: dict(value) for key, value in self._inflight.items()}
