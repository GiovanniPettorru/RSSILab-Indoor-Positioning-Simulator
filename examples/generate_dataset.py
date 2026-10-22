"""Generate an RSS radio-map dataset (CSV) for training / testing positioning methods.

    python examples/generate_dataset.py      (from the repository folder)

Columns: Coord_X, Coord_Y, Anc1_RSS ... AncN_RSS (one row per reading); readings
below the receiver sensitivity are written as nan. With the reference scenario,
the legacy preset, 1 m spacing and 10000 samples per point this reproduces the
layout and the channel of the original RSSILab dataset (61 x 31 points); here a
smaller number of samples keeps the file small.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from a clone

from rssilab import engine, presets                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

scn = presets.default_scenario()                         # 90 x 60 m room, 60 x 30 m area, 12 anchors
tech = presets.technology("Legacy log-normal model")     # P0 -40 dBm, n 4, sigma 3 dB, legacy fading
path = os.path.join(OUT, "dataset_legacy_1m_100samples.csv")
rows = engine.export_dataset_csv(path, scn, tech, spacing=1.0, samples=100, seed=1)
print(f"{rows:,} rows -> {path}")

# The same grid with a realistic BLE channel in the demo office (walls, correlated shadowing),
# skipping the grid points that fall inside walls and racks
office = presets.demo_office()
path = os.path.join(OUT, "dataset_ble_office_1m_20samples.csv")
rows = engine.export_dataset_csv(path, office, presets.technology("BLE 2.4 GHz"), spacing=1.0,
                                 samples=20, seed=1, skip_obstacles=True)
print(f"{rows:,} rows -> {path}")
