"""Repository-relative locations of the maps and configs (scenarios, suites)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
MAPS_DIR = ROOT / "maps"
SCENARIOS_DIR = CONFIGS_DIR / "scenarios"
SUITES_DIR = CONFIGS_DIR / "suites"
