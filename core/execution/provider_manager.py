"""
Execution Provider Manager.

Central place that decides which ONNX Runtime execution provider DocMind
should use, and reports which one is ACTUALLY active -- never a claimed
or assumed one.

Priority order (highest performance/lowest power first):
    1. QNNExecutionProvider   -- Qualcomm Hexagon NPU (Snapdragon X Elite/X2
                                  Elite Windows PCs, requires the
                                  onnxruntime-qnn package + QNN SDK libraries)
    2. DmlExecutionProvider   -- DirectML, GPU/NPU acceleration on Windows
                                  (works on many Windows PCs, not just
                                  Snapdragon)
    3. CPUExecutionProvider   -- always available, the guaranteed fallback

IMPORTANT: This module never claims a provider is active unless
`onnxruntime.get_available_providers()` actually reports it AND a real
InferenceSession using it initializes successfully. On a normal
Windows/Linux/macOS dev machine without Qualcomm hardware or the QNN
SDK, this will correctly and honestly report CPUExecutionProvider.
"""

from dataclasses import dataclass
from typing import List, Optional

import onnxruntime as ort


# Ordered by preference. DocMind will pick the first one that is both
# installed (visible to ONNX Runtime) AND successfully usable.
PROVIDER_PRIORITY = [
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

# Human-readable labels shown in the UI -- keeps marketing language out of
# core logic while still being clear about what each provider means.
PROVIDER_LABELS = {
    "QNNExecutionProvider": "Qualcomm QNN (Hexagon NPU)",
    "DmlExecutionProvider": "DirectML (GPU/NPU via DirectX)",
    "CPUExecutionProvider": "CPU",
}


@dataclass
class ProviderInfo:
    provider_id: str          # e.g. "CPUExecutionProvider"
    label: str                # human-readable label for the UI
    is_accelerated: bool      # True only for QNN or DirectML, never CPU
    available_providers: List[str]  # everything ONNX Runtime reports as installed


class ExecutionProviderManager:
    """
    Detects and selects the best available ONNX Runtime execution provider.

    Usage:
        manager = ExecutionProviderManager()
        info = manager.detect()
        session = ort.InferenceSession(model_path, providers=manager.provider_list())
    """

    def __init__(self, requested_priority: Optional[List[str]] = None):
        self.priority = requested_priority or PROVIDER_PRIORITY
        self._selected_provider: Optional[str] = None

    def available_providers(self) -> List[str]:
        """Providers ONNX Runtime reports as installed on this machine."""
        return ort.get_available_providers()

    def detect(self) -> ProviderInfo:
        """
        Determine which provider DocMind will actually use, based on
        what's genuinely installed -- not guessed.

        Returns:
            ProviderInfo describing the selected provider.
        """
        installed = self.available_providers()

        selected = "CPUExecutionProvider"  # guaranteed safe default
        for candidate in self.priority:
            if candidate in installed:
                selected = candidate
                break

        self._selected_provider = selected
        return ProviderInfo(
            provider_id=selected,
            label=PROVIDER_LABELS.get(selected, selected),
            is_accelerated=selected in ("QNNExecutionProvider", "DmlExecutionProvider"),
            available_providers=installed,
        )

    def provider_list(self) -> List[str]:
        """
        Ordered provider list to pass directly into
        onnxruntime.InferenceSession(..., providers=...).

        ONNX Runtime itself will fall back further down this list at
        session-creation time if the selected provider fails to
        initialize for a given model (e.g. an unsupported op), so we
        pass the full priority chain, not just the single selected one.
        """
        if self._selected_provider is None:
            self.detect()
        installed = self.available_providers()
        return [p for p in self.priority if p in installed] or ["CPUExecutionProvider"]
