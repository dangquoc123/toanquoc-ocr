"""Table reconstruction (Design §4): exact morphology + SLANet fallback.

:func:`reconstruct_grid` (morphology) needs only numpy + opencv.  :class:`SLANet`
additionally needs torch and is imported lazily to keep the morphology path
dependency-light.
"""

from .morphology import Cell, TableGrid, extract_rules, reconstruct_grid

__all__ = ["Cell", "TableGrid", "extract_rules", "reconstruct_grid", "SLANet"]


def __getattr__(name):  # lazy torch import for SLANet
    if name == "SLANet":
        from .slanet import SLANet
        return SLANet
    raise AttributeError(name)
