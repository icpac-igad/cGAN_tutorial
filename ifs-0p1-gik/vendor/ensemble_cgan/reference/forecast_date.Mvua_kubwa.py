#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Same as forecast.py, but the date to process is given as a command line argument

# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_fcst_fields

import os
import sys
import pathlib
import yaml
from argparse import ArgumentParser

import netCDF4 as nc
import numpy as np
from tensorflow.keras.utils import Progbar

from data import (
    HOURS,
    all_fcst_fields,
    accumulated_fields,
    nonnegative_fields,
    fcst_norm,
    logprec,
    denormalise,
    load_hires_constants,
)
import read_config
from noise import NoiseGenerator
from setupmodel import setup_model

from datetime import datetime, timedelta


def run_gen_forecast(input_file: str):

    # Define the latitude and longitude arrays for later
    latitude = np.arange(-13.65, 24.7, 0.1)
    longitude = np.arange(19.15, 54.3, 0.1)

    # Some setup
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
    downscaling_steps = read_config.read_downscaling_factor()["steps"]
    assert fcst_norm is not None

    # In[3]:

    # Open and parse forecast.yaml
    # fcstyaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast.yaml")
    fcstyaml_path = "forecast.yaml"
    with open(fcstyaml_path, "r") as f:
        try:
            fcst_params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    # In[4]:

    model_folder = fcst_params["MODEL"]["folder"]
    checkpoint = fcst_params["MODEL"]["checkpoint"]
    input_folder = fcst_params["INPUT"]["folder"]

    # input_file = fcst_params["INPUT"]["file"]
    # Instead of reading input_file from forecast.yaml, get it from the command line
    # input_file = f"IFS_{time_str}_00Z.nc"

    start_hour = fcst_params["INPUT"]["start_hour"]
    end_hour = fcst_params["INPUT"]["end_hour"]
    output_folder = fcst_params["OUTPUT"]["folder"]
    ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]

    assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
    assert end_hour % HOURS == 0, f"end_hour must be divisible by {HOURS}"

    # Open and parse GAN config file
    config_path = os.path.join(model_folder, "setup_params.yaml")
    with open(config_path, "r") as f:
        try:
            setup_params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    mode = setup_params["GENERAL"]["mode"]
    arch = setup_params["MODEL"]["architecture"]
    padding = setup_params["MODEL"]["padding"]
    filters_gen = setup_params["GENERATOR"]["filters_gen"]
    noise_channels = setup_params["GENERATOR"]["noise_channels"]
    latent_variables = setup_params["GENERATOR"]["latent_variables"]
    filters_disc = setup_params["DISCRIMINATOR"][
        "filters_disc"
    ]  # TODO: avoid setting up discriminator in forecast mode?
    constant_fields = 2

    assert (
        mode == "GAN"
    ), "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

    # Set up pre-trained GAN
    weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
    input_channels = 2 * len(all_fcst_fields)

    model = setup_model(
        mode=mode,
        arch=arch,
        downscaling_steps=downscaling_steps,
        input_channels=input_channels,
        constant_fields=constant_fields,
        filters_gen=filters_gen,
        filters_disc=filters_disc,
        noise_channels=noise_channels,
        latent_variables=latent_variables,
        padding=padding,
    )
    gen = model.gen
    gen.load_weights(weights_fn)

    network_const_input = load_hires_constants(batch_size=1)  # 1 x lats x lons x 2

    # In[5]:

    def create_output_file(nc_out_path):
        netcdf_dict = {}
        rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
        netcdf_dict["rootgrp"] = rootgrp
        rootgrp.description = (
            "GAN 24-hour rainfall ensemble members in the ICPAC region."
        )

        # Create output file dimensions
        rootgrp.createDimension("latitude", len(latitude))
        rootgrp.createDimension("longitude", len(longitude))
        rootgrp.createDimension("member", ensemble_members)
        rootgrp.createDimension("time", None)
        rootgrp.createDimension("valid_time", None)

        # Create variables
        latitude_data = rootgrp.createVariable("latitude", "f4", ("latitude",))
        latitude_data.units = "degrees_north"
        latitude_data[:] = latitude  # Write the latitude data

        longitude_data = rootgrp.createVariable("longitude", "f4", ("longitude",))
        longitude_data.units = "degrees_east"
        longitude_data[:] = longitude  # Write the longitude data

        ensemble_data = rootgrp.createVariable("member", "i4", ("member",))
        ensemble_data.units = "ensemble member"
        ensemble_data[:] = range(1, ensemble_members + 1)

        netcdf_dict["time_data"] = rootgrp.createVariable("time", "f4", ("time",))
        netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

        netcdf_dict["valid_time_data"] = rootgrp.createVariable(
            "fcst_valid_time", "f4", ("time", "valid_time")
        )
        netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

        netcdf_dict["precipitation"] = rootgrp.createVariable(
            "precipitation",
            "f4",
            ("time", "member", "valid_time", "latitude", "longitude"),
            compression="zlib",
            chunksizes=(1, 1, 1, len(latitude), len(longitude)),
        )
        netcdf_dict["precipitation"].units = "mm/h"
        netcdf_dict["precipitation"].long_name = "Precipitation"

        return netcdf_dict

    # In[6]:

    # Open input netCDF file to get the times
    nc_in_path = os.path.join(input_folder, input_file)
    nc_in = nc.Dataset(nc_in_path, mode="r")
    start_times = nc_in["time"][:]
    valid_times = nc_in["valid_time"][:]
    # nc_in.close()

    # The datetime corresponding to this start time
    d = datetime(1900, 1, 1) + timedelta(hours=int(start_times[0]))

    # Create output netCDF file
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    nc_out_path = os.path.join(
        output_folder, f"GAN_{d.year}{d.month:02d}{d.day:02d}_00Z.nc"
    )
    netcdf_dict = create_output_file(nc_out_path)
    netcdf_dict["time_data"][0] = start_times[0]

    # copy across valid_time from input file
    # For 7x 24h forecasts with lead times of 6, 30, 54, 78, 102, 126, 150 hours
    in_time_idx = ([1, 5, 9, 13, 17, 21, 25],)
    valid_times_forecast = valid_times[in_time_idx]
    netcdf_dict["valid_time_data"][0, :] = valid_times_forecast

    # For each valid time
    for valid_time_num in range(len(valid_times_forecast)):

        # the contents of the next loop are v. similar to load_fcst from data.py,
        # but not quite the same, since that has different assumptions on how the
        # forecast data is stored.  TODO: unify the data normalisation between these?
        field_arrays = []
        for field in all_fcst_fields:
            # Original:
            # nc_in[field] has shape 1 x 50 x 29 x 384 x 352
            # corresponding to n_forecasts x n_ensemble_members x n_valid_times x n_lats x n_lons
            # Ensemble mean:
            # nc_in[field] has shape len(nc_in["time"]) x 29 x 384 x 352

            # Open input netCDF file
            # input_file = f"{field}.nc"
            # nc_in_path = os.path.join(input_folder, input_file)
            # nc_in = nc.Dataset(nc_in_path, mode="r")
            all_data_mean = nc_in[f"{field}_ensemble_mean"]
            all_data_sd = nc_in[f"{field}_ensemble_standard_deviation"]

            if field in accumulated_fields:
                # For a 24h forecast, and a 6h lead time
                # data1 = np.mean(all_data_mean[idx, 1:5, :, :], axis=0)            # Mean of the accumulations
                # data2 = np.sqrt(np.mean(all_data_sd[idx, 1:5, :, :]**2, axis=0))  # RMS of the standard deviations
                # For a 24h forecast, and a 30h lead time
                # data1 = np.mean(all_data_mean[idx, 5:9, :, :], axis=0)            # Mean of the accumulations
                # data2 = np.sqrt(np.mean(all_data_sd[idx, 5:9, :, :]**2, axis=0))  # RMS of the standard deviations
                # For a particular valid time
                data1 = np.mean(
                    all_data_mean[
                        valid_time_num * 4 + 1 : valid_time_num * 4 + 5, :, :
                    ],
                    axis=0,
                )  # Mean of the accumulations
                data2 = np.sqrt(
                    np.mean(
                        all_data_sd[
                            valid_time_num * 4 + 1 : valid_time_num * 4 + 5, :, :
                        ]
                        ** 2,
                        axis=0,
                    )
                )  # RMS of the standard deviations
                data = np.stack([data1, data2], axis=-1)

            else:
                # return mean and std computed using the trapezium rule
                # For a 24h forecast, and a 6h lead time
                # temp_data_mean = all_data_mean[idx, 1:6, :, :]
                # temp_data_var = all_data_sd[idx, 1:6, :, :]**2  # Convert to variances
                # For a 24h forecast, and a 30h lead time
                # temp_data_mean = all_data_mean[idx, 5:10, :, :]
                # temp_data_var = all_data_sd[idx, 5:10, :, :]**2  # Convert to variances
                # For a particular valid time
                temp_data_mean = all_data_mean[
                    valid_time_num * 4 + 1 : valid_time_num * 4 + 6, :, :
                ]
                temp_data_var = (
                    all_data_sd[valid_time_num * 4 + 1 : valid_time_num * 4 + 6, :, :]
                    ** 2
                )  # Convert to variances

                data1 = (
                    temp_data_mean[0, :, :] / 2
                    + np.sum(temp_data_mean[1:4, :, :], axis=0)
                    + temp_data_mean[4, :, :] / 2
                ) / 4
                data2 = (
                    temp_data_var[0, :, :] / 2
                    + np.sum(temp_data_var[1:4, :, :], axis=0)
                    + temp_data_var[4, :, :] / 2
                ) / 4
                data = np.stack([data1, np.sqrt(data2)], axis=-1)

            # perform normalisation on forecast data
            if field in nonnegative_fields:
                data = np.maximum(
                    data, 0.0
                )  # eliminate any data weirdness/regridding issues

            if field in ["tp", "cp"]:
                # precip is measured in metres, so multiply to get mm
                data *= 1000
                data /= HOURS  # convert to mm/hr
            elif field in accumulated_fields:
                # for all other accumulated fields [just ssr for us]
                data /= (
                    HOURS * 3600
                )  # convert from a 6-hr difference to a per-second rate

            if field in ["tp", "cp"]:
                data = logprec(data, True)
            else:
                # apply transformation to make fields O(1), based on historical
                # forecast data from one of the training years
                if field in ["mcc"]:
                    # already 0-1
                    pass
                elif field in ["sp", "t2m"]:
                    # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
                    data[:, :, 0] -= fcst_norm[field]["mean"]
                    data /= fcst_norm[field]["std"]
                elif field in nonnegative_fields:
                    data /= fcst_norm[field]["max"]
                else:
                    # winds
                    data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])

            field_arrays.append(data)

        network_fcst_input = np.concatenate(
            field_arrays, axis=-1
        )  # lat x lon x 2*len(all_fcst_fields)
        network_fcst_input = np.expand_dims(
            network_fcst_input, axis=0
        )  # 1 x lat x lon x 2*len(...)

        noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=1)
        progbar = Progbar(ensemble_members)
        for ii in range(ensemble_members):
            gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
            gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
            netcdf_dict["precipitation"][0, ii, valid_time_num, :, :] = denormalise(
                gan_prediction[0, :, :, 0]
            )
            progbar.add(1)

    netcdf_dict["rootgrp"].close()

    # Close the ECMWF forecasts NetCDF file
    nc_in.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="entrypoint script for cGAN forecasts",
        usage="python forecast_date.py date_string <in format YYYYMMDD>",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        dest="input_file",
        help="relative path to IFS forecast file as defined on data_paths.yml config",
    )
    args = parser.parse_args()
    run_gen_forecast(args.input_file)
