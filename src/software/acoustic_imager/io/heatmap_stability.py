"""Runtime application of heatmap stability presets (FPS HUD)."""

from __future__ import annotations


def apply_heatmap_stability_preset(index: int) -> None:
    """Set config attributes for preset 0=Sharp, 1=Balanced, 2=Smooth."""
    from acoustic_imager import config

    presets = getattr(config, "HEATMAP_STABILITY_PRESETS", None)
    if presets is None:
        return
    if not (0 <= index < len(presets)):
        return
    for k, v in presets[index].items():
        setattr(config, k, v)
    config.HEATMAP_STABILITY_PRESET_INDEX = index
