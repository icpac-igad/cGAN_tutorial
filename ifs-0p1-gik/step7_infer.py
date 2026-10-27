"""STEP 7 -- cGAN inference on one IFS_<YYYYMMDD>_<HH>Z.nc from step 6 (--frame tf).

The same computation as scripts/ensemble-forecast.py, made runnable from here:
paths come from arguments instead of forecast.yaml / data_paths.yaml, the input
is checked against the model's frame before anything runs, and the members are
predicted in batches instead of one by one.

Per 6-hour window [t, t+6 h], for each of the 14 fields (data.all_fcst_fields):
  accumulated (tp cp ssr):  mean and sd over members of (end - start), 0, 0
  others:                   mean, sd at start; mean, sd at end
then the normalisation of ensemble-forecast.py lines 204-233 (FCSTNorm2018.pkl),
-> 1 x 384 x 352 x 56, + elev/lsm constants, + noise -> generator -> mm h-1.

Output GAN_IFS_<YYYYMMDD>_<HH>Z.nc:
  precipitation (time, member, valid_time, latitude, longitude)  mm h-1
  fcst_valid_time (time, valid_time)  the START of each 6-hour window, as
  ensemble-forecast.py writes it (hours since 1900-01-01)

Needs TensorFlow (requirements-infer.txt) and, from the trained model:
  --model-folder  with setup_params.yaml and models/gen_weights-<checkpoint>.h5
  --constants     with elev.nc, lsm.nc, FCSTNorm2018.pkl (the 384 x 352 frame)

  python step7_infer.py --nc nc/IFS_20261021_00Z.nc --model-folder /path/ICPAC-big-ensmeansd \\
      --checkpoint 316800 --constants /path/CONSTANTS --outdir gan
  python step7_infer.py --nc nc/IFS_20261021_00Z.nc --constants /path/CONSTANTS --dry-run

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
TF_CODE = os.path.normpath(os.path.join(HERE, "..", "tensorflow-dev-test"))

# data.py (tensorflow-dev-test/data/data.py) -- copied, not imported: importing
# it reads data_paths.yaml and the CONSTANTS_PATH of whichever site is set there
HOURS = 6
ALL_FCST_FIELDS = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw", "tcrw",
                   "tcw", "tcwv", "tp", "u700", "v700"]
ACCUMULATED = ["cp", "ssr", "tp"]
NONNEGATIVE = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw", "tcrw", "tcw",
               "tcwv", "tp"]
FRAME = (384, 352)


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


def window_input(nc_in, t, norm):
    """Network input for the window [valid_time t, t+1]: 1 x 384 x 352 x 56."""
    arrays = []
    for field in ALL_FCST_FIELDS:
        temp = np.ma.filled(nc_in[f"{field}_ensemble_mean"][0, :, t:t + 2, :, :], np.nan)
        start, end = temp[:, 0], temp[:, 1]
        if field in ACCUMULATED:
            diff = end - start
            mean, std = diff.mean(axis=0), diff.std(axis=0, ddof=1)
            zeros = np.zeros(mean.shape)
            data = np.stack([mean, std, zeros, zeros], axis=-1)
        else:
            data = np.stack([start.mean(axis=0), start.std(axis=0, ddof=1),
                             end.mean(axis=0), end.std(axis=0, ddof=1)], axis=-1)
        if field in NONNEGATIVE:
            data = np.maximum(data, 0.0)
        if field in ["tp", "cp"]:
            data = logprec(data * 1000 / HOURS)            # m per 6 h -> mm h-1, log
        else:
            if field in ACCUMULATED:                        # ssr: J m-2 per 6 h -> W m-2
                data = data / (HOURS * 3600)
            if field == "mcc":
                pass
            elif field in ["sp", "t2m"]:
                data[:, :, 0] -= norm[field]["mean"]
                data[:, :, 2] -= norm[field]["mean"]
                data /= norm[field]["std"]
            elif field in NONNEGATIVE:
                data /= norm[field]["max"]
            else:
                data /= max(-norm[field]["min"], norm[field]["max"])
        arrays.append(data)
    x = np.concatenate(arrays, axis=-1)[np.newaxis].astype(np.float32)
    if not np.isfinite(x).all():
        sys.exit(f"window {t}: non-finite network input")
    return x


def load_generator_code(tf_code):
    """generator() from <tf_code>/model/models.py and NoiseGenerator from
    model/noise.py, WITHOUT model/__init__.py: that imports the training stack
    (gan.py, vaegantrain.py -> evaluation -> cartopy, data_paths.yaml).

    layers.py also imports keras.layers._Merge, which Keras 2.15 no longer
    exports at the top level (it is the base of RandomWeightedAverage, a
    training-only layer): it is aliased to where 2.15 keeps it.
    """
    import importlib
    import types
    import keras.layers as kl
    if not hasattr(kl, "_Merge"):
        from keras.src.layers.merging.base_merge import _Merge
        kl._Merge = _Merge
    pkg = types.ModuleType("cgan_tf_model")
    pkg.__path__ = [os.path.join(tf_code, "model")]
    sys.modules["cgan_tf_model"] = pkg
    models = importlib.import_module("cgan_tf_model.models")
    noise = importlib.import_module("cgan_tf_model.noise")
    return models.generator, noise.NoiseGenerator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", required=True, help="IFS_<YYYYMMDD>_<HH>Z.nc from step 6 --frame tf")
    ap.add_argument("--constants", required=True, help="dir with elev.nc lsm.nc FCSTNorm2018.pkl")
    ap.add_argument("--model-folder", help="dir with setup_params.yaml and models/")
    ap.add_argument("--checkpoint", type=int, help="e.g. 316800 -> gen_weights-0316800.h5")
    ap.add_argument("--tf-code", default=TF_CODE, help="repo dir holding setupmodel.py and model/")
    ap.add_argument("--start-hour", type=int, default=0)
    ap.add_argument("--end-hour", type=int, default=None,
                    help="default: the last valid time in the file (168 h; 144 h for 06z/18z)")
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
    shape = nc_in["tp_ensemble_mean"].shape                 # 1 x member x vt x lat x lon
    if tuple(shape[-2:]) != FRAME:
        sys.exit(f"{a.nc} is {shape[-2]} x {shape[-1]}, the model needs {FRAME}: "
                 f"make it with step6_make_cgan_netcdf.py --frame tf")
    lat = nc_in["latitude"][:]
    if not (abs(lat[0] + 13.65) < 1e-3 and abs(float(nc_in["longitude"][0]) - 19.15) < 1e-3):
        sys.exit("latitude/longitude are not the TF frame (-13.65N.., 19.15E..)")
    n_vt = shape[2]
    end_hour = a.end_hour if a.end_hour is not None else (n_vt - 1) * HOURS
    assert a.start_hour % HOURS == 0 and end_hour % HOURS == 0
    if end_hour // HOURS > n_vt - 1:
        sys.exit(f"end_hour {end_hour} beyond the file's last valid time {(n_vt - 1) * HOURS} h")
    windows = list(range(a.start_hour // HOURS, end_hour // HOURS))
    const, norm = load_constants(a.constants)
    if const.shape[1:3] != FRAME:
        sys.exit(f"constants are {const.shape[1:3]}, not {FRAME}")
    print(f"input    {a.nc}  {shape[1]} members x {n_vt} valid times  -> "
          f"{len(windows)} windows {a.start_hour}..{end_hour} h")

    if a.dry_run:
        t0 = time.time()
        for t in windows:
            x = window_input(nc_in, t, norm)
            if t in (windows[0], windows[-1]):
                print(f"  window {t * HOURS:3d}-{(t + 1) * HOURS:3d} h  input {x.shape}  "
                      f"range {x.min():.3g}..{x.max():.3g}  tp mean ch "
                      f"{x[..., 4 * ALL_FCST_FIELDS.index('tp')].mean():.3g}")
        print(f"dry run  OK: {len(windows)} windows of finite 1x384x352x56 input "
              f"in {time.time() - t0:.0f}s")
        return

    if not (a.model_folder and (a.checkpoint or a.untrained)):
        sys.exit("--model-folder and --checkpoint are needed (or --dry-run)")
    import yaml
    gen_fn, NoiseGenerator = load_generator_code(a.tf_code)
    with open(os.path.join(a.model_folder, "setup_params.yaml")) as f:
        sp = yaml.safe_load(f)
    with open(os.path.join(a.tf_code, "config", "downscaling_factor.yaml")) as f:
        steps = yaml.safe_load(f)["steps"]
    assert sp["GENERAL"]["mode"] == "GAN", "only GAN models"
    # as setupmodel.setup_model(mode="GAN") builds it; the discriminator and
    # the WGANGP wrapper are training-only and not built here
    gen = gen_fn(mode="GAN", arch=sp["MODEL"]["architecture"], downscaling_steps=steps,
                 input_channels=4 * len(ALL_FCST_FIELDS), constant_fields=2,
                 filters_gen=sp["GENERATOR"]["filters_gen"],
                 noise_channels=sp["GENERATOR"]["noise_channels"],
                 padding=sp["MODEL"]["padding"])
    if a.untrained:
        wfn = "UNTRAINED (random weights, test only)"
    else:
        wfn = os.path.join(a.model_folder, "models", f"gen_weights-{a.checkpoint:07}.h5")
        gen.load_weights(wfn)
    print(f"model    {wfn}")

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"GAN_{os.path.basename(a.nc)}")
    tmp = out + ".part"
    root = nc.Dataset(tmp, "w", format="NETCDF4")
    root.description = "GAN 6-hour rainfall ensemble members in the ICPAC region."
    root.input_file = os.path.basename(a.nc)
    root.model = f"{wfn}"
    for k in ("source_store", "source_snapshot"):
        if k in nc_in.ncattrs():
            root.setncattr(f"input_{k}", nc_in.getncattr(k))
    root.createDimension("latitude", FRAME[0])
    root.createDimension("longitude", FRAME[1])
    root.createDimension("member", a.ensemble_members)
    root.createDimension("time", None)
    root.createDimension("valid_time", None)
    for name, src in (("latitude", "latitude"), ("longitude", "longitude")):
        v = root.createVariable(name, "f4", (name,))
        v.units, v[:] = nc_in[src].units, nc_in[src][:]
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
    for k, t in enumerate(windows):
        vt[0, k] = nc_in["valid_time"][t]
        x = window_input(nc_in, t, norm)
        done = 0
        while done < a.ensemble_members:
            b = min(a.batch, a.ensemble_members - done)
            y = gen.predict([np.repeat(x, b, 0), np.repeat(const, b, 0), noise_gen()[:b]],
                            verbose=False)
            pr[0, done:done + b, k, :, :] = denormalise(y[..., 0])
            done += b
        print(f"  window {t * HOURS:3d}-{(t + 1) * HOURS:3d} h  {a.ensemble_members} members  "
              f"{time.time() - t0:6.0f}s", flush=True)
    root.close()
    nc_in.close()
    os.replace(tmp, out)
    print(f"wrote {out}  {os.path.getsize(out) / 1e9:.2f} GB  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
