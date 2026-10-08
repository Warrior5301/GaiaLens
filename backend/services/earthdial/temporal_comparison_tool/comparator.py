"""
EarthDial Temporal Comparison Tool (Facade)
===========================================

This module maintains backward compatibility for existing scripts while delegating
to the modular `TemporalComparisonTool` in `tool.py`.
"""

from tool import TemporalComparisonTool

# Backward-compatible alias
EarthDialTemporalComparator = TemporalComparisonTool

__all__ = ["EarthDialTemporalComparator", "TemporalComparisonTool"]
