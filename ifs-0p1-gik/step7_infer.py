"""STEP 7 -- cGAN inference on one IFS_<YYYYMMDD>_<HH>Z.nc.

Three input layouts, told apart by the file itself (global attr `cgan_product`):

  members   step6_make_cgan_netcdf.py --frame tf: every member, mean/sd taken
            here; the computation of scripts/ensemble-forecast.py. 6 h windows
            from --start-hour to --end-hour (default: the whole file).
  jurre-6h  step6_sewaa_netcdf.py: the SEWAA 6 h model, as
            ensemble-cgan@Jurre_brishti dsrnngan/test_forecast.py. The file
            holds +30..54 h, so 4 windows 30-36 .. 48-54 h. 56 input channels.
  mvua-7d   step6_sewaa_netcdf.py: the SEWAA 24 h model, as
            ensemble-cgan@Mvua_kubwa dsrnngan/forecast_date.py. 7 days of 24 h
            starting +6, 30, .., 150 h, so the last ends at +174 h. 28 channels.
(vendor/PROVENANCE.md has the source commits; vendor/ensemble_cgan/reference/
holds the two operational scripts verbatim.)

Per window and field, the 4 (or 2) channels are built as the matching script
builds them, then the same normalisation (FCSTNorm2018.pkl), + elev/lsm
constants, + noise -> generator -> mm h-1. Paths are arguments instead of
forecast.yaml / data_paths.yaml, the input frame is checked first, and the
members are predicted in batches instead of one by one.

The generator is built from the vendored ensemble-cgan code (vendor/ensemble_cgan,
models.py etc., byte-identical to both operational branches).

Output (layout of the original scripts):
  precipitation (time, member, valid_time, latitude, longitude)  mm h-1
  fcst_valid_time (time, valid_time)  the START of each window (hours since 1900-01-01)
  members:  <outdir>/GAN_IFS_<YYYYMMDD>_<HH>Z.nc
  jurre-6h, mvua-7d: <outdir>/GAN_<YYYYMMDD>_<HH>Z.nc  (the SEWAA name)

Needs TensorFlow 2.15 (requirements-infer.txt) and, from the trained model:
  --model-folder  with setup_params.yaml and models/gen_weights-<checkpoint>.h5
  --constants     with elev.nc, lsm.nc, FCSTNorm2018.pkl (the 384 x 352 frame)

  python step7_infer.py --nc nc/cgan-ifs-6h-ens/IFS_20261026_00Z.nc \\
      --model-folder /path/jurre-brishti/model-weights --checkpoint 316800 \\
      --constants /path/jurre-brishti/constants --outdir gan/jurre-brishti-ens
  python step7_infer.py --nc nc/cgan-ifs-7d-ens/IFS_20261026_00Z.nc --constants ... --dry-run

--dry-run builds and checks every network input without TensorFlow or weights.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_CODE = os.path.join(HERE, "vendor", "ensemble_cgan")

# data.py (ensemble-cgan dsrnngan/data.py) -- copied, not imported: importing
# it reads data_paths.yaml and the CONSTANTS_PATH of whichever site is set there
HOURS = 6
ALL_FCST_FIELDS = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw", "tcrw",
                   "tcw", "tcwv", "tp", "u700", "v700"]
ACCUMULATED = ["cp", "ssr", "tp"]
NONNEGATIVE = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw", "tcrw", "tcw",
               "tcwv", "tp"]
FRAME = (384, 352)
# channels per field, and which of them are means (sp/t2m have the climatological
# mean subtracted from those only, "but NOT from ens sd")
CHANNELS = {"members": (4, (0, 2)), "jurre-6h": (4, (0, 2)), "mvua-7d": (2, (0,))}


def denormalise(x):
    return np.minimum(10 ** x - 1.0, 100.0)


def logprec(y):
    return np.log10(1.0 + y)


def load_constants(cdir):
    import xarray as xr
    z = xr.load_dataset(os.path.join(cdir, "elev.nc"))["elevation"].values / 10000.0
    lsm = xr.load_dataset(os.path.join(cdir, "lsm.nc"))["lsm"].values
    with open(os.path.join(cdir, "FCSTNorm2018.pkl"), "rb") as f:
        norm = pickle.load(f)
    const = np.stack([z, lsm], axis=-1)[np.newaxis].astype(np.float32)   # 1 x H x W x 2
    return const, norm


def normalise(field, data, norm, mean_channels):
    """The normalisation block shared by all three scripts (lat x lon x channels)."""
    if field in NONNEGATIVE:
        data = np.maximum(data, 0.0)
    if field in ["tp", "cp"]:
        return logprec(data * 1000 / HOURS)                # m per 6 h -> mm h-1, log
    if field in ACCUMULATED:                                # ssr: J m-2 per 6 h -> W m-2
        data = data / (HOURS * 3600)
    if field == "mcc":
        pass
    elif field in ["sp", "t2m"]:
        for c in mean_channels:
            data[:, :, c] -= norm[field]["mean"]
        data /= norm[field]["std"]
    elif field in NONNEGATIVE:
        data /= norm[field]["max"]
    else:
        data /= max(-norm[field]["min"], norm[field]["max"])
    return data


def members_input(nc_in, t, field):
    """ensemble-forecast.py: window [t, t+1] from every member."""
    temp = np.ma.filled(nc_in[f"{field}_ensemble_mean"][0, :, t:t + 2, :, :], np.nan)
    start, end = temp[:, 0], temp[:, 1]
    if field in ACCUMULATED:
        diff = end - start
        mean, std = diff.mean(axis=0), diff.std(axis=0, ddof=1)
        zeros = np.zeros(mean.shape)
        return np.stack([mean, std, zeros, zeros], axis=-1)
    return np.stack([start.mean(axis=0), start.std(axis=0, ddof=1),
                     end.mean(axis=0), end.std(axis=0, ddof=1)], axis=-1)


def jurre_input(nc_in, t, field):
    """test_forecast.py: window [t, t+1]; accumulations are pre-differenced at t."""
    m = np.ma.filled(nc_in[f"{field}_ensemble_mean"][t:t + 2], np.nan).astype(np.float64)
    s = np.ma.filled(nc_in[f"{field}_ensemble_standard_deviation"][t:t + 2],
                     np.nan).astype(np.float64)
    if field in ACCUMULATED:
        zeros = np.zeros(m[0].shape)
        return np.stack([m[0], s[0], zeros, zeros], axis=-1)
    return np.stack([m[0], s[0], m[1], s[1]], axis=-1)


def mvua_input(nc_in, d, field):
    """forecast_date.py: day d spans indices 4d+1 .. 4d+5 (+6+24d .. +30+24d h)."""
    i = 4 * d + 1
    m = np.ma.filled(nc_in[f"{field}_ensemble_mean"][i:i + 5], np.nan).astype(np.float64)
    s = np.ma.filled(nc_in[f"{field}_ensemble_standard_deviation"][i:i + 5],
                     np.nan).astype(np.float64)
    if field in ACCUMULATED:                    # mean of the 4 increments, RMS of their sds
        return np.stack([m[:4].mean(axis=0), np.sqrt((s[:4] ** 2).mean(axis=0))], axis=-1)
    v = s ** 2                                  # trapezium rule over the 5 instants
    trap = lambda x: (x[0] / 2 + x[1:4].sum(axis=0) + x[4] / 2) / 4  # noqa: E731
    return np.stack([trap(m), np.sqrt(trap(v))], axis=-1)


BUILD = {"members": members_input, "jurre-6h": jurre_input, "mvua-7d": mvua_input}


def network_input(nc_in, product, w, norm):
    """1 x 384 x 352 x (56 | 28) for window / day w."""
    _, mean_ch = CHANNELS[product]
    x = np.concatenate([normalise(f, BUILD[product](nc_in, w, f), norm, mean_ch)
                        for f in ALL_FCST_FIELDS], axis=-1)[np.newaxis].astype(np.float32)
    if not np.isfinite(x).all():
        sys.exit(f"window {w}: non-finite network input")
    return x


def windows_of(nc_in, product, start_hour, end_hour):
    """(window indices, index of each window's start valid time, label of each)."""
    n_vt = len(nc_in["valid_time"])
    vt = np.asarray(nc_in["valid_time"][:], dtype=np.float64)
    lead = [int(round(v)) for v in vt - float(nc_in["time"][0])]
    if product == "members":
        start = max(start_hour, lead[0])          # hours -> indices from the file's first
        end = end_hour if end_hour is not None else lead[-1]
        assert start % HOURS == 0 and end % HOURS == 0
        if end > lead[-1]:
            sys.exit(f"end_hour {end} beyond the file's last valid time {lead[-1]} h")
        w = list(range((start - lead[0]) // HOURS, (end - lead[0]) // HOURS))
        return w, w, [f"{lead[t]:3d}-{lead[t + 1]:3d} h" for t in w]
    if product == "jurre-6h":
        if lead[0] != 30:
            sys.exit(f"jurre-6h input must start at +30 h (test_forecast.py), not +{lead[0]} h")
        w = list(range(n_vt - 1))
        return w, w, [f"{lead[t]:3d}-{lead[t + 1]:3d} h" for t in w]
    if lead[0] != 0 or n_vt < 30 or lead[29] != 174:
        sys.exit(f"mvua-7d input must hold +0..174 h every 6 h, not {lead[0]}..{lead[-1]} h")
    w = list(range(7))
    return w, [4 * d + 1 for d in w], [f"{lead[4 * d + 1]:3d}-{lead[4 * d + 5]:3d} h" for d in w]


def load_generator_code(gen_code):
    """generator() and NoiseGenerator from the vendored ensemble-cgan files.
    Flat modules (`from blocks import ...`), so the folder goes on sys.path."""
    import importlib
    sys.path.insert(0, gen_code)
    models = importlib.import_module("models")
    noise = importlib.import_module("noise")
    return models.generator, noise.NoiseGenerator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", required=True, help="IFS_<YYYYMMDD>_<HH>Z.nc from step 6")
    ap.add_argument("--constants", required=True, help="dir with elev.nc lsm.nc FCSTNorm2018.pkl")
    ap.add_argument("--model-folder", help="dir with setup_params.yaml and models/")
    ap.add_argument("--checkpoint", type=int, help="e.g. 316800 -> gen_weights-0316800.h5")
    ap.add_argument("--gen-code", default=GEN_CODE,
                    help="dir holding models.py blocks.py layers.py noise.py "
                         "downscaling_factor.yaml (default: vendor/ensemble_cgan)")
    ap.add_argument("--start-hour", type=int, default=0, help="members layout only")
    ap.add_argument("--end-hour", type=int, default=None,
                    help="members layout only; default: the last valid time in the file")
    ap.add_argument("--ensemble-members", type=int, default=50, help="GAN samples per window")
    ap.add_argument("--batch", type=int, default=10, help="GAN samples per predict() call")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", default="gan")
    ap.add_argument("--untrained", action="store_true",
                    help="TEST ONLY: build the generator from setup_params.yaml but skip "
                         "the weights, to exercise the inference path; output is noise")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check every input; no TensorFlow, no weights")
    a = ap.parse_args()

    import netCDF4 as nc
    nc_in = nc.Dataset(a.nc, "r")
    product = (nc_in.getncattr("cgan_product") if "cgan_product" in nc_in.ncattrs()
               else "members")
    if product not in CHANNELS:
        sys.exit(f"unknown cgan_product {product!r} in {a.nc}")
    shape = nc_in["tp_ensemble_mean"].shape
    if tuple(shape[-2:]) != FRAME:
        sys.exit(f"{a.nc} is {shape[-2]} x {shape[-1]}, the model needs {FRAME}: "
                 f"make it with step6 on the TF frame")
    lat = nc_in["latitude"][:]
    if not (abs(lat[0] + 13.65) < 1e-3 and abs(float(nc_in["longitude"][0]) - 19.15) < 1e-3):
        sys.exit("latitude/longitude are not the TF frame (-13.65N.., 19.15E..)")
    windows, vt_idx, labels = windows_of(nc_in, product, a.start_hour, a.end_hour)
    n_ch = CHANNELS[product][0] * len(ALL_FCST_FIELDS)
    const, norm = load_constants(a.constants)
    if const.shape[1:3] != FRAME:
        sys.exit(f"constants are {const.shape[1:3]}, not {FRAME}")
    print(f"input    {a.nc}  {product} layout, {len(nc_in['valid_time'])} valid times  -> "
          f"{len(windows)} windows {labels[0].split('-')[0].strip()}..{labels[-1].split('-')[1]}")

    if a.dry_run:
        t0 = time.time()
        for w, lab in zip(windows, labels):
            x = network_input(nc_in, product, w, norm)
            if w in (windows[0], windows[-1]):
                print(f"  window {lab}  input {x.shape}  range {x.min():.3g}..{x.max():.3g}  "
                      f"tp mean ch {x[..., n_ch // 14 * ALL_FCST_FIELDS.index('tp')].mean():.3g}")
        print(f"dry run  OK: {len(windows)} windows of finite 1x384x352x{n_ch} input "
              f"in {time.time() - t0:.0f}s")
        return

    if not (a.model_folder and (a.checkpoint or a.untrained)):
        sys.exit("--model-folder and --checkpoint are needed (or --dry-run)")
    import yaml
    gen_fn, NoiseGenerator = load_generator_code(a.gen_code)
    with open(os.path.join(a.model_folder, "setup_params.yaml")) as f:
        sp = yaml.safe_load(f)
    with open(os.path.join(a.gen_code, "downscaling_factor.yaml")) as f:
        steps = yaml.safe_load(f)["steps"]
    assert sp["GENERAL"]["mode"] == "GAN", "only GAN models"
    # as setupmodel.setup_model(mode="GAN") builds it; the discriminator and
    # the WGANGP wrapper are training-only and not built here
    gen = gen_fn(mode="GAN", arch=sp["MODEL"]["architecture"], downscaling_steps=steps,
                 input_channels=n_ch, constant_fields=2,
                 filters_gen=sp["GENERATOR"]["filters_gen"],
                 noise_channels=sp["GENERATOR"]["noise_channels"],
                 padding=sp["MODEL"]["padding"])
    if a.untrained:
        wfn = "UNTRAINED (random weights, test only)"
    else:
        wfn = os.path.join(a.model_folder, "models", f"gen_weights-{a.checkpoint:07}.h5")
        gen.load_weights(wfn)
    print(f"model    {wfn}  ({n_ch} input channels)")

    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.basename(a.nc)
    out = os.path.join(a.outdir, f"GAN_{base}" if product == "members"
                       else "GAN_" + base.replace("IFS_", "", 1))
    tmp = out + ".part"
    root = nc.Dataset(tmp, "w", format="NETCDF4")
    root.description = (f"GAN {'24' if product == 'mvua-7d' else '6'}-hour rainfall "
                        "ensemble members in the ICPAC region.")
    root.input_file = base
    root.cgan_product = product
    root.model = f"{wfn}"
    for k in ("source_store", "source_snapshot"):
        if k in nc_in.ncattrs():
            root.setncattr(f"input_{k}", nc_in.getncattr(k))
    root.createDimension("latitude", FRAME[0])
    root.createDimension("longitude", FRAME[1])
    root.createDimension("member", a.ensemble_members)
    root.createDimension("time", None)
    root.createDimension("valid_time", None)
    for name in ("latitude", "longitude"):
        v = root.createVariable(name, "f4", (name,))
        v.units, v[:] = nc_in[name].units, nc_in[name][:]
    v = root.createVariable("member", "i4", ("member",))
    v.units, v[:] = "ensemble member", range(1, a.ensemble_members + 1)
    tvar = root.createVariable("time", "f4", ("time",))
    tvar.units = "hours since 1900-01-01 00:00:00.0"
    tvar[0] = nc_in["time"][0]
    vt = root.createVariable("fcst_valid_time", "f4", ("time", "valid_time"))
    vt.units = "hours since 1900-01-01 00:00:00.0"
    pr = root.createVariable("precipitation", "f4", ("time", "member", "valid_time",
                                                     "latitude", "longitude"),
                             compression="zlib", chunksizes=(1, 1, 1) + FRAME)
    pr.units, pr.long_name = "mm h**-1", "Precipitation"

    t0 = time.time()
    noise_gen = NoiseGenerator(FRAME + (sp["GENERATOR"]["noise_channels"],),
                               batch_size=a.batch, random_seed=a.seed)
    for k, (w, vi, lab) in enumerate(zip(windows, vt_idx, labels)):
        vt[0, k] = nc_in["valid_time"][vi]
        x = network_input(nc_in, product, w, norm)
        done = 0
        while done < a.ensemble_members:
            b = min(a.batch, a.ensemble_members - done)
            y = gen.predict([np.repeat(x, b, 0), np.repeat(const, b, 0), noise_gen()[:b]],
                            verbose=False)
            pr[0, done:done + b, k, :, :] = denormalise(y[..., 0])
            done += b
        print(f"  window {lab}  {a.ensemble_members} members  {time.time() - t0:6.0f}s",
              flush=True)
    root.close()
    nc_in.close()
    os.replace(tmp, out)
    print(f"wrote {out}  {os.path.getsize(out) / 1e9:.2f} GB  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
