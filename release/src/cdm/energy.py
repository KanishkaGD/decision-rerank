"""
GPU energy of a run, read from NVML's cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, millijoules
since the driver was loaded; Volta and newer). Called through ctypes on libnvidia-ml.so.1, which `apptainer --nv`
binds into the container, so no Python package is needed.

The counter belongs to the device, not the process: the reading covers everything on the GPU the job was given,
which is only our process. It covers the GPU board only -- not CPU, memory or cooling. The device is found by the
UUID PyTorch reports for cuda:0, so it is the right one even when NVML sees every GPU of the node.
"""

import ctypes
from typing import Any, Dict, Optional


class GpuEnergyMeter:
    """meter.start() ... meter.stop() -> joules on the job's GPU, or None if there is no GPU or no NVML."""

    def __init__(self):
        self._nvml = None
        self._handle = None
        self._start: Optional[int] = None
        try:
            import torch

            if not torch.cuda.is_available():
                return
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            if nvml.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            uuid_bytes = (uuid if uuid.startswith("GPU-") else f"GPU-{uuid}").encode()
            if nvml.nvmlDeviceGetHandleByUUID(ctypes.c_char_p(uuid_bytes), ctypes.byref(handle)) != 0:
                if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                    return
            self._nvml, self._handle = nvml, handle
            if self._read() is None:  # counter not supported (pre-Volta)
                self._nvml = None
        except (OSError, AttributeError, ImportError):
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def _read(self) -> Optional[int]:
        value = ctypes.c_ulonglong()
        if self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(value)) != 0:
            return None
        return value.value

    def start(self) -> None:
        self._start = self._read() if self.available else None

    def stop(self) -> Optional[float]:
        if not self.available or self._start is None:
            return None
        end = self._read()
        return None if end is None else (end - self._start) / 1000.0

    def describe(self) -> Dict[str, Any]:
        return {"energy_counter": "nvmlDeviceGetTotalEnergyConsumption" if self.available else None}
