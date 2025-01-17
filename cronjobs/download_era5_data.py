"""
Script that updates existing ERA5 reanalysis datasets stored in the S3 bucket: era5-reanalysis
"""

import os
import glob
import boto3
import cdsapi
import zipfile
import datetime
import numpy as np
import xarray as xr
from shutil import rmtree
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made to host ")

# Data directory where downloads are stored temporarily for processing
DATA_DIR_ERA5 = ""

# Name of the S3 buckets where current CAMS analysis datasets are stored
S3_BUCKET_ERA5 = "era5-reanalysis"

# Credentials for S3 client
S3_ACCESS_KEY = ""
S3_SECRET_KEY = ""
S3_ENDPOINT = "https://obs.eu-de.otc.t-systems.com"

# Credentials for the Climate Data Store
# For more information, see: https://cds.climate.copernicus.eu/api-how-to
CDS_URL = ""
CDS_KEY = ""

AREAS = {  # extent of the area of interest for which the data should be selected
    "ireland": [
        55.65, -11.35, 51.35,
        -5.25,
    ],
}

DATASETS = {
    "reanalysis-era5-single-levels":{
        "PARAMS": {
            "format": "netcdf",
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "total_precipitation",
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00",
            ],
            "area": AREAS["ireland"],
            "grid":[0.1,0.1],
        },
        "FILE_NAME":"era5"
    },
    # "reanalysis-era5-land":{
    #     "PARAMS": {
    #         "format": "netcdf",
    #         "product_type": "reanalysis",
    #         "variable": [
    #             "10m_u_component_of_wind",
    #             "10m_v_component_of_wind",
    #             "2m_temperature",
    #             "total_precipitation",
    #         ],
    #         "time": [
    #             "00:00", "01:00", "02:00",
    #             "03:00", "04:00", "05:00",
    #             "06:00", "07:00", "08:00",
    #             "09:00", "10:00", "11:00",
    #             "12:00", "13:00", "14:00",
    #             "15:00", "16:00", "17:00",
    #             "18:00", "19:00", "20:00",
    #             "21:00", "22:00", "23:00",
    #         ],
    #         "area": AREAS["ireland"],
    #     },
    #     "FILE_NAME":"era5_land"
    # },
}


def _calc_time_difference(start_date):
    """The ADS api allows up to 5000 fields per request. This function calculates the no. of hours between the
    current date minus two days and the last date for which data is present in the CAMS dataset. In case the difference
    in hours is over 5000 the request must be split into two different requests"""

    end_date = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=7)
    difference = (end_date - start_date).days * 24  # difference between start and end date in hours

    return difference


def _check_attrs(ds):
    """
    Checks for each of the data variables in the Dataset, if any of the original attributes is missing. If so, adds
    the attributes again
    """

    attrs_dict = {  # dictionary containing the attribute information to add per variable
        'u10':  # variable name
            {'units': 'm s**-1', 'long_name': '10 metre U wind component'},  # attributes
        'v10':
            {'units': 'm s**-1', 'long_name': '10 metre V wind component'},
        'tp':
            {'units': 'm', 'long_name': 'Total precipitation'},
        't2m':
            {'units': 'K', 'long_name': '2 metre temperature'},
    }

    for var in list(ds.data_vars):  # loop through the data variables
        if 'units' not in ds[var].attrs.keys():  # if the attribute is not present, add it
            ds[var].attrs['units'] = attrs_dict[var]['units']

        if 'long_name' not in ds[var].attrs.keys():
            ds[var].attrs['long_name'] = attrs_dict[var]['long_name']

    return ds


def _calc_download_periods(last_date, hour_interval=1226):
    """
    The CDS api allows up to 5000 fields per request. In case the difference in hours is over 5000 the download must
    be split into multiple requests. This function generates a dictionary containing the start and end date for each
    download, while making sure that the number of fields for the request do not exceed 5000 hours.

    :param last_date: datetime obj. containing the last date for which data is available in the downloaded CAMS product
    :param hour_interval: the number of hours/fields to allow for each request, for era5 we request 4 variables at once
    so hour interval chosen here is 1226 (5000/4 - 24).
    :return: dictionary containing the start and end dates for each api download call
    """

    # Calculate the time difference in hours
    time_difference = _calc_time_difference(start_date=last_date)
    # Calculate how many download periods are needed
    no_downloads = int(np.ceil(time_difference/hour_interval))

    download_periods = {}  # empty dict to store the dates for each download

    if no_downloads == 1:  # In case only a single download is needed

        # Set the end data to today minus seven days - ERA5 data takes a while before it is available via the API
        end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=7)

        download_periods[f"download_1"] = {  # add the start and end date info to the dict
            "start": datetime.date(
                year=last_date.year,
                month=last_date.month,
                day=last_date.day,
            ),
            "end": datetime.date(
                year=end.year,
                month=end.month,
                day=end.day,
            )
        }

        return download_periods

    for d in range(1, (no_downloads + 1)):  # in case of multiple downloads

        if not (d == no_downloads):   # if not the last download

            if d == 1:   # first download starting from the last available date for the existing dataset
                start = last_date
                end = start + timedelta(hours=hour_interval)
            else:  # not the first or last download, so we can use the information already in the download periods dict
                start = download_periods[f"download_{d-1}"]["end"] + timedelta(days=1)
                end = start + timedelta(hours=hour_interval)

        else:  # for the last download the final date will be less than the indicated hour interval
            start = download_periods[f"download_{d-1}"]["end"] + timedelta(days=1)
            end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=7)

        download_periods[f"download_{d}"] = {  # add the start and end date info to the dict
            "start": datetime.date(
                year=start.year,
                month=start.month,
                day=start.day,
            ),
            "end": datetime.date(
                year=end.year,
                month=end.month,
                day=end.day,
            )
        }

    return download_periods


def _zip_nc_file(directory, nc_file, name):
    """
    Function to zip a netcdf file in the directory where the nc file is stored

    :param directory: path to the directory where the nc file to be zipped is located
    :param nc_file: name of the nc file to be zipped
    :param name: name for the zipfile
    """

    current_dir = os.getcwd()  # get the current working directory
    os.chdir(directory)  # change directory to the directory where to file we want to zip is located

    # Create the compressed file
    zipfile.ZipFile(f"{name}.zip", mode="w").write(
        f"{nc_file}.nc",
        compress_type=zipfile.ZIP_DEFLATED
    )

    os.chdir(current_dir)  # change back to the original working directory


def main():
    """Main Function - code to download the CAMS data join it with newly downloaded data and upload to a S3 bucket"""
    for dataset in DATASETS:

        # directory for files downloaded using the CDS api
        if not os.path.exists(f"{DATA_DIR_ERA5}/temp/api"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_ERA5}/temp/api", mode=0o777)

        # directory for files downloaded from the S3 bucket
        if not os.path.exists(f"{DATA_DIR_ERA5}/temp/s3"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_ERA5}/temp/s3", mode=0o777)

        # Load the existing netcdf containing the CAMS data
        dataset_name = DATASETS[dataset]["FILE_NAME"]

        s3_client = boto3.client(  # initiate S3 client
            "s3",
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            endpoint_url=S3_ENDPOINT
        )

        s3_client.download_file(  # download the dataset currently stored in the S3 bucket
            S3_BUCKET_ERA5,
            f"{dataset_name}.zip",
            f"{DATA_DIR_ERA5}/temp/s3/{dataset_name}.zip"
        )

        # unzip the file downloaded from the s3 bucket
        with zipfile.ZipFile(f"{DATA_DIR_ERA5}/temp/s3/{dataset_name}.zip", "r") as zip_ref:
            zip_ref.extractall(f"{DATA_DIR_ERA5}/temp/s3/")

        # load & copy dataset
        existing_pollutant_dataset = xr.load_dataset(f"{DATA_DIR_ERA5}/temp/s3/{dataset_name}.nc")
        existing_pollutant_dataset_cp = existing_pollutant_dataset.copy(deep=True)

        # close the dataset and remove from memory, work only with the copy from now on
        existing_pollutant_dataset.close()
        existing_pollutant_dataset = None

        # Get the most recent date in the existing CAMS netcdf as a datetime object
        last_date = datetime.datetime.strptime(
            np.datetime_as_string(
                existing_pollutant_dataset_cp.time[-1].values,
                unit="s"
            ),
            "%Y-%m-%dT%H:%M:%S"
        )

        # We want to start downloading one day after the last date
        last_date += timedelta(days=1)

        # calculate how many download requests we have to make
        download_periods = _calc_download_periods(last_date)

        for period in download_periods:

            start_date = download_periods[period]["start"].strftime("%Y-%m-%d")
            end_date = download_periods[period]["end"].strftime("%Y-%m-%d")

            new_dataset_filename = f"{DATA_DIR_ERA5}/temp/api/{dataset_name}_{start_date}_{end_date}.nc"

            # additional parameters, will be different for each download
            additional_params = {
                "date": f"{start_date}/{end_date}",
            }

            # combine the additional parameters with the existing parameters into a single dict
            params = {**DATASETS[dataset]["PARAMS"], **additional_params}

            if not os.path.exists(new_dataset_filename):
                try:
                    c = cdsapi.Client(
                        url=CDS_URL,
                        key=CDS_KEY
                    )  # Initiate the API
                    c.retrieve(  # make the api call
                        dataset,
                        params,
                        new_dataset_filename
                    )
                except Exception as e:
                    print(  # in case of an error
                        f"The API request for {dataset} over the period {start_date} to {end_date}"
                        f" gave the following error:\n{e}")

        new_data = [  # open all the newly downloaded datasets and convert the time column to datetime
            xr.open_dataset(f) for f in glob.glob(f"{DATA_DIR_ERA5}/temp/api/{dataset_name}*.nc")
        ]

        if dataset == "reanalysis-era5-single-levels":
            # ERA5 has an extra dimension "expver" as as result of data from the ERA5T (near real time) dataset being
            # used before the reanalysis data become available. The following line gets rid of this extra dimension if
            # present:

            new_data = [
                ds.sel(expver=1).combine_first(ds.sel(expver=5)) if "expver" in list(ds.dims) else ds for ds in new_data
            ]

        # create list of all the datasets to combine into the new dataset
        all_data = [existing_pollutant_dataset_cp] + new_data  # create list of all the datasets to combine

        # combine the datasets into single dataset
        updated_dataset = xr.combine_nested(all_data, concat_dim="time")
        # check the attribute information
        updated_dataset = _check_attrs(updated_dataset)

        # Write the backup file & updated file as netcdf files
        existing_pollutant_dataset_cp.to_netcdf(f"{DATA_DIR_ERA5}/temp/{dataset_name}_backup.nc")
        updated_dataset.to_netcdf(f"{DATA_DIR_ERA5}/temp/{dataset_name}.nc")

        # zip the new nc file and the backup nc file
        _zip_nc_file(f"{DATA_DIR_ERA5}/temp/", f"{dataset_name}_backup", f"{dataset_name}_backup")
        _zip_nc_file(f"{DATA_DIR_ERA5}/temp/", dataset_name, dataset_name)

        # upload the compressed folders to s3 buckets, overwriting the existing objects
        s3_client.upload_file(
            f"{DATA_DIR_ERA5}/temp/{dataset_name}_backup.zip",
            S3_BUCKET_ERA5,
            f"{dataset_name}_backup.zip"
        )

        s3_client.upload_file(
            f"{DATA_DIR_ERA5}/temp/{dataset_name}.zip",
            S3_BUCKET_ERA5,
            f"{dataset_name}.zip"
        )

        # clear variables from memory

        existing_pollutant_dataset_cp = None
        updated_dataset = None
        new_data = None
        all_data = None

        # Finally, remove all the downloaded files from the temp folder
        rmtree(f"{DATA_DIR_ERA5}/temp/")


if __name__ == "__main__":
    main()
