import json
from pathlib import Path

import numpy as np
import pytest

from ligotools import readligo as rl


@pytest.fixture(scope="module")
def event_meta():
    """
    Load the first available event from data/BBH_events_v3.json and resolve files.
    We only parametrize detectors whose files actually exist, so CI doesn't fail
    if a file is missing.
    """
    data_dir = Path("data")
    json_path = data_dir / "BBH_events_v3.json"
    if not json_path.exists():
        pytest.skip(f"Missing {json_path}; HW data files not present.")

    with open(json_path, "r") as f:
        events = json.load(f)

    if "GW150914" in events:
        ev = events["GW150914"]
    else:
        ev = next(iter(events.values()))

    meta = {
        "fs": float(ev["fs"]),
        "H1": data_dir / ev["fn_H1"],
        "L1": data_dir / ev["fn_L1"],
    }

    available_dets = []
    for det in ("H1", "L1"):
        if meta[det].exists():
            available_dets.append(det)

    if not available_dets:
        pytest.skip("No H1/L1 data files found in data/; cannot test loaddata.")

    meta["available_dets"] = tuple(available_dets)
    return meta


# ---------- Tests ----------

@pytest.mark.parametrize("det", ["H1", "L1"])
def test_loaddata_basic_contract(event_meta, det):
    """loaddata returns (strain, time, channel_dict) with sensible shapes and keys."""
    if det not in event_meta["available_dets"]:
        pytest.skip(f"{det} file not present; skipping.")

    fname = str(event_meta[det])
    strain, time, chan = rl.loaddata(fname, det)

    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan, dict)

    # non-empty + aligned shapes
    assert strain.size > 0
    assert time.size == strain.size

    dt_arr = np.diff(time)
    assert np.all(dt_arr > 0), "time must increase"

    assert "DATA" in chan, "Expected 'DATA' mask in channel dict"
    assert chan["DATA"].size > 0
    assert "DEFAULT" in chan, "Expected 'DEFAULT' channel to be set"


@pytest.mark.parametrize("det", ["H1", "L1"])
def test_sampling_interval_matches_ts(event_meta, det):
    """
    The median time step should be ~ 1/fs from metadata.
    (read_hdf5 uses the 'Xspacing' attribute; we check consistency.)
    """
    if det not in event_meta["available_dets"]:
        pytest.skip(f"{det} file not present; skipping.")

    fs = event_meta["fs"]
    fname = str(event_meta[det])
    _, time, _ = rl.loaddata(fname, det)

    dt_med = float(np.median(np.diff(time)))
    expected = 1.0 / fs
    assert np.isclose(dt_med, expected, rtol=1e-6, atol=1e-9), f"dt={dt_med}, expected≈{expected}"

def test_loaddata_bad_path_returns_none_tuple():
    """
    readligo.loaddata returns (None, None, None) for missing files per its implementation
    (os.stat(...) guarded by try/except).
    """
    strain, time, dq = rl.loaddata("data/THIS_FILE_DOES_NOT_EXIST.hdf5", "H1")
    assert strain is None and time is None and dq is None