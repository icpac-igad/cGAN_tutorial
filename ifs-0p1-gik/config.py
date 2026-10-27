"""Everything that defines WHAT is built. Edit here, nowhere else.

Read by the client-side scripts (step1..step6). The worker module
`cgan_dag.py` does NOT import this file: the box and field table travel to the
workers as task arguments, so changing the box needs no worker redeploy.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Domain: East Africa, the cGAN inference box. Every edge lies on the 0.1 deg
# grid (-180 + 0.1k, 90 - 0.1j), so the cut is exact -- no interpolation.
#   401 rows (25N -> 15S, store order descending) x 331 cols (20E -> 53E)
# NOTE: the TF cGAN trained frame is 384 x 352 at -13.65..24.65N,
# 19.15..54.25E (cGAN_tutorial/docs/ecmwf_icechunk_dask_variable_extraction.md
# section 5.2). 20..53E does not cover its east/west edges -- README section 8.
# ---------------------------------------------------------------------------
BOX = dict(lat_max=25.0, lat_min=-15.0, lon_min=20.0, lon_max=53.0, deg=0.1)
BOX_TAG = "20e53e-25n15s"

# ---------------------------------------------------------------------------
# The 14 cGAN forecast fields (data.all_fcst_fields; RFE2_cGAN_RUNBOOK.md):
#   cgan name -> (.index param, level hPa or None, virtual-store array name)
# The .index param drives the prefix read; the store array name drives the
# virtual-store read used for checks. `cape` is published as `mucape` since
# 2024-11-13 (same channel to the model); `t2m` is `2t` in the .index.
# Accumulated fields (tp, cp, ssr) are stored as published -- accumulated from
# step 0 -- and the cGAN inference differences them itself.
# ---------------------------------------------------------------------------
FIELDS = {
    "cape": ("mucape", None, "mucape"),
    "cp":   ("cp",     None, "cp"),
    "mcc":  ("mcc",    None, "mcc"),
    "sp":   ("sp",     None, "sp"),
    "ssr":  ("ssr",    None, "ssr"),
    "t2m":  ("2t",     None, "t2m"),
    "tciw": ("tciw",   None, "tciw"),
    "tclw": ("tclw",   None, "tclw"),
    "tcrw": ("tcrw",   None, "tcrw"),
    "tcw":  ("tcw",    None, "tcw"),
    "tcwv": ("tcwv",   None, "tcwv"),
    "tp":   ("tp",     None, "tp"),
    "u700": ("u",      700,  "u"),
    "v700": ("v",      700,  "v"),
}
ACCUMULATED = ("tp", "cp", "ssr")

# 0..168 h every 6 h = 29 lead times: what ensemble-forecast.py indexes as
# valid_time (HOURS = 6). All are on both the 3-hourly (<=144 h) and the
# 6-hourly (>144 h) part of the 00z/12z step axis.
STEPS = list(range(0, 169, 6))
MEMBERS = list(range(51))          # 0 = control (oper/fc), 1..50 perturbed

ERA = "0p1"

# ---------------------------------------------------------------------------
# Where things live. A store "spec" is one of
#   /abs/path  or  local:/abs/path        a directory on this machine
#   ceph:<prefix>                         EWC Ceph, bucket must-icechunk
#   sourcecoop:<prefix>                   source.coop, bucket e4drr-project
# (stores.py). Override any of these with the env var of the same name.
# ---------------------------------------------------------------------------
LOCAL_ROOT = os.environ.get("IFS0P1_LOCAL_ROOT",
                            os.path.expanduser("~/icechunk-local"))

# Virtual store (references only, ~13 MB per cycle). The public one already
# holds every 0.1 deg cycle from 20261020 06z that was appended.
VD_LOCAL = os.environ.get("VD_LOCAL", f"{LOCAL_ROOT}/ecmwf-ifs-0p1-icechunk-vd")
VD_SOURCECOOP = "forecasts/ecmwf-ifs-0p1-icechunk-vd"

# Realized store: ONE store per cycle, so retention is a prefix delete.
REAL_LOCAL_ROOT = os.environ.get("REAL_LOCAL_ROOT",
                                 f"{LOCAL_ROOT}/ecmwf-ifs-0p1-cgan14-{BOX_TAG}")
REAL_CEPH_ROOT = f"cgan14-{BOX_TAG}"
REAL_SOURCECOOP_ROOT = f"forecasts/ecmwf-ifs-0p1-cgan14-{BOX_TAG}"


def cycle_name(date8: str, run: str = "00") -> str:
    return f"{date8}{run}z"


def real_spec(mode: str, date8: str, run: str = "00") -> str:
    """Realized-store spec for one cycle under `mode` (local|ceph|sourcecoop)."""
    c = cycle_name(date8, run)
    return {"local": f"local:{REAL_LOCAL_ROOT}/{c}",
            "ceph": f"ceph:{REAL_CEPH_ROOT}/{c}",
            "sourcecoop": f"sourcecoop:{REAL_SOURCECOOP_ROOT}/{c}"}[mode]


def vd_spec(mode: str) -> str:
    return {"local": f"local:{VD_LOCAL}",
            "sourcecoop": f"sourcecoop:{VD_SOURCECOOP}"}[mode]


def box_shape(box=BOX):
    ny = int(round((box["lat_max"] - box["lat_min"]) / box["deg"])) + 1
    nx = int(round((box["lon_max"] - box["lon_min"]) / box["deg"])) + 1
    return ny, nx
