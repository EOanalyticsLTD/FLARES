""""
Script that updates existing GFAS FRP and wildfire flux datasets stored in the S3 bucket: gfas
"""

import os
import glob
import boto3
import zipfile
import calendar
import datetime
import numpy as np
import xarray as xr
from shutil import rmtree
from datetime import timedelta
from dateutil import relativedelta
from ecmwfapi import ECMWFDataServer

import warnings
# Filters out a warning that pops up when calling the ECMWF api when using the credentials as parameters
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made to host ")

# Data directory where downloads are stored temporarily for processing
DATA_DIR_GFAS = ""

# Name of the S3 buckets where current GFAS datasets are stored
S3_BUCKET_GFAS = "gfas"

# Credentials for S3 client
S3_ACCESS_KEY = ""
S3_SECRET_KEY = ""
S3_ENDPOINT = "https://obs.eu-de.otc.t-systems.com"

# Credentials for the ECMWF API
# For more information, see: https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api

# IMPORTANT:
# ECMWF API key is valid for one year. Additionaly if it is your first time downloading a dataset via the API, you need
# to make sure that you have accepted the terms and conditions for that particular dataset, otherwise the API will give
# an error.

ECMWF_URL = "https://api.ecmwf.int/v1"
ECMWF_KEY = ""
ECMWF_EMAIL = ""


AREAS = {  # extent of the area of interest for which the data should be selected
    "ireland": "55.65/-11.35/51.35/-5.25",
}

# GFAS datasets to download. FRP is available on an hourly basis, however the wildfire flux datasets are only daily
# averages

# the parameter "param" indicates which variable to download
# the list below gives the specific information for which number represents which variable.

# 99.210 : Wildfire radiative power
# 80.210 : Wildfire flux of Carbon Dioxide
# 81.210 : Wildfire flux of Carbon Monoxide
# 82.210 : Wildfire flux of Methane
# 85.210 : Wildfire flux of Nitrogen Oxides NOx
# 87.210 : Wildfire flux of Particulate Matter PM2.5
# 88.210 : Wildfire flux of Total Particulate Matter
# 102.210: Wildfire flux of Sulfur Dioxide

# For the parameter "type" there are two options:

# gsd : gridded satellite data (hourly) - Only available for fire radiative power (FRP)\
# ga : gridded average (daily average)


DATASETS = {
    "frp":{
        "PARAMS":{
            "class": "mc",
            "dataset": "cams_gfas",
            "expver": "0001",
            "levtype": "sfc",
            "param": "99.210",  # see list above
            "step": "0-24",
            "stream": "gfas",
            "time": "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23",
            "type": "gsd",
            "ident": "784/783",  # 784: Aqua, 783: Terra
            "instrument": "389",  # 389: MODIS
            "area": AREAS["ireland"],
            "format": "netcdf",
            "grid":"0.1/0.1",
        },
        "FILE_NAME":"frp"
    },
    "wildfire_flux":{
        "PARAMS":{
            "class": "mc",
            "dataset": "cams_gfas",
            "expver": "0001",
            "levtype": "sfc",
            "param": "80.210/81.210/82.210/85.210/87.210/88.210/102.210",
            "step": "0-24",
            "stream": "gfas",
            "time": "00",
            "type": "ga",
            "area": AREAS["ireland"],
            "format": "netcdf",
            "grid":"0.1/0.1",
        },
        "FILE_NAME":"wildfire_flux"
    }
}


def _calc_download_periods(last_date):
    """
    From the documentation:

    To retrieve data efficiently (and get your data quicker!) you should retrieve all the data you need from one tape,
    then from the next tape, and so on. For GFAS, this means retrieving all the data you need for one month,
    then for the next month, and so on.

    This function gets the number of downloads needed, when downloading the data per month.

    :param last_date: datetime obj. containing the last date for which data is available in the downloaded CAMS product
    :return: dictionary containing the start and end dates for each api download call
    """

    end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=2)
    # get the time difference between the start and end date
    difference = relativedelta.relativedelta(end, last_date)

    # in case there is a positive difference in days and hours other than months -> add 1 to difference in months
    if difference.hours > 0 or difference.days > 0:
        months_difference = difference.months + 1
    else:
        months_difference = difference.months

    months_difference += difference.years * 12  # make sure to account for difference in years

    download_periods = {}  # empty dict to store the dates for each download

    if months_difference == 1:  # only one download

        download_periods[f"download_1"] = {
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
    else:  # in case of multiple downloads
        for month in range(1, (months_difference + 1)):
            if not (month == months_difference):  # if not the last download
                if month == 1:  # if first download
                    start = last_date
                    days_in_month = calendar.monthrange(last_date.year, int(last_date.month))[1]
                    end = last_date.replace(day=days_in_month)
                else:
                    start = download_periods[f"download_{month - 1}"]["end"] + timedelta(days=1)
                    days_in_month = calendar.monthrange(start.year, int(start.month))[1]
                    end = start.replace(day=days_in_month)

            else:  # last download, final date not for this download should be the end date
                start = download_periods[f"download_{month - 1}"]["end"] + timedelta(days=1)
                end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=2)

            download_periods[f"download_{month}"] = {
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
    """Main Function - code to download the CAMS data join it with newly downloaded data and upload it to a S3 bucket"""
    for dataset in DATASETS:

        # directory for files downloaded using the ECMWF api
        if not os.path.exists(f"{DATA_DIR_GFAS}/temp/api"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_GFAS}/temp/api", mode=0o777)

        # directory for files downloaded from the S3 bucket
        if not os.path.exists(f"{DATA_DIR_GFAS}/temp/s3"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_GFAS}/temp/s3", mode=0o777)

        # Load the existing netcdf containing the CAMS data
        dataset_name = DATASETS[dataset]["FILE_NAME"]

        s3_client = boto3.client(  # initiate S3 client
            "s3",
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            endpoint_url=S3_ENDPOINT
        )

        s3_client.download_file(  # download the dataset currently stored in the S3 bucket
            S3_BUCKET_GFAS,
            f"{dataset_name}.zip",
            f"{DATA_DIR_GFAS}/temp/s3/{dataset_name}.zip"
        )

        # unzip the file downloaded from the s3 bucket
        with zipfile.ZipFile(f"{DATA_DIR_GFAS}/temp/s3/{dataset_name}.zip", "r") as zip_ref:
            zip_ref.extractall(f"{DATA_DIR_GFAS}/temp/s3/")

        # load & copy dataset
        existing_pollutant_dataset = xr.load_dataset(f"{DATA_DIR_GFAS}/temp/s3/{dataset_name}.nc")
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

        # calculate how many download requests we have to make, for the GFAS data requests are best split up per month
        download_periods = _calc_download_periods(last_date)

        for period in download_periods:

            start_date = download_periods[period]["start"].strftime("%Y-%m-%d")
            end_date = download_periods[period]["end"].strftime("%Y-%m-%d")

            # path to location where we want to store the download
            new_dataset_filename = f"{DATA_DIR_GFAS}/temp/api/{dataset_name}_{start_date}_{end_date}.nc"

            # additional parameters, will be different for each download
            additional_params = {
                "date": f"{start_date}/to/{end_date}",
                "target": new_dataset_filename,
            }

            # combine the additional parameters with the existing parameters into a single dict
            params = {**DATASETS[dataset]["PARAMS"], **additional_params}

            if not os.path.exists(new_dataset_filename):
                try:
                    c = ECMWFDataServer(
                        url=ECMWF_URL,
                        key=ECMWF_KEY,
                        email=ECMWF_EMAIL,
                    )  # Initiate the API client
                    c.retrieve(  # make the api call
                        params
                    )
                except Exception as e:
                    print(  # in case of an error
                        f"The API request for {dataset} over the period {start_date} to {end_date}"
                        f"gave the following error:\n{e}")
                    raise

        new_data = [  # open all the newly downloaded datasets and convert the time column to datetime
            xr.open_dataset(f) for f in glob.glob(f"{DATA_DIR_GFAS}/temp/api/{dataset_name}*.nc")
        ]

        all_data = [existing_pollutant_dataset_cp] + new_data  # create list of all the datasets to combine

        # combine the datasets
        updated_dataset = xr.combine_nested(all_data, concat_dim="time", combine_attrs='override')

        # Convert the data to 64 bit float, otherwise lower values are rounded to 0
        updated_dataset = updated_dataset.astype(np.float64)

        # Write the backup file & updated file as netcdf files
        existing_pollutant_dataset_cp.to_netcdf(f"{DATA_DIR_GFAS}/temp/{dataset_name}_backup.nc")
        updated_dataset.to_netcdf(f"{DATA_DIR_GFAS}/temp/{dataset_name}.nc")

        # zip the new nc file and the backup nc file
        _zip_nc_file(f"{DATA_DIR_GFAS}/temp/", f"{dataset_name}_backup", f"{dataset_name}_backup")
        _zip_nc_file(f"{DATA_DIR_GFAS}/temp/", dataset_name, dataset_name)

        # upload the compressed folders to s3 buckets, overwriting the existing objects
        s3_client.upload_file(
            f"{DATA_DIR_GFAS}/temp/{dataset_name}_backup.zip",
            S3_BUCKET_GFAS,
            f"{dataset_name}_backup.zip"
        )

        s3_client.upload_file(
            f"{DATA_DIR_GFAS}/temp/{dataset_name}.zip",
            S3_BUCKET_GFAS,
            f"{dataset_name}.zip"
        )

        # Finally, remove all the downloaded files from the temp folder
        existing_pollutant_dataset_cp = None
        updated_dataset = None
        new_data = None
        all_data = None

        rmtree(f"{DATA_DIR_GFAS}/temp/")


if __name__ == "__main__":
    main()
