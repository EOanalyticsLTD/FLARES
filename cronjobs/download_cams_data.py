"""
Script that updates existing CAMS analysis datasets stored in the S3 bucket: cams-analysis
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

# Filters out a warning that pops up when calling the ADS api when using the credentials as parameters
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made to host ")

POLLUTANTS = {  # pollutants for which data should be acquired from the api
    "carbon_monoxide": {  # parameter string for the API call
        "cams_var_name":"co_conc"  # variable name / netcdf file name
    },
    "nitrogen_dioxide": {
        "cams_var_name":"no2_conc"
    },
    "nitrogen_monoxide": {
        "cams_var_name":"no_conc"
    },
    "ozone": {
        "cams_var_name":"o3_conc"
    },
    "particulate_matter_10um": {
        "cams_var_name":"pm10_conc"
    },
    "particulate_matter_2.5um":{
        "cams_var_name":"pm2p5_conc"
    },
    "sulphur_dioxide": {
        "cams_var_name":"so2_conc"
    },
    "pm10_wildfires":{
        "cams_var_name":"pmwf_conc"
    }
}

# Data directory where downloads are stored temporarily for processing
DATA_DIR_CAMS = ""

# Name of the S3 buckets where current CAMS analysis datasets are stored
S3_BUCKET_CAMS = "cams-analysis"

# Credentials for S3 client
S3_ACCESS_KEY = ""
S3_SECRET_KEY = ""
S3_ENDPOINT = "https://obs.eu-de.otc.t-systems.com"

# Credentials for the Atmosphere Data Store
# For more information, see: https://ads.atmosphere.copernicus.eu/api-how-to
ADS_URL = "https://ads.atmosphere.copernicus.eu/api/v2"
ADS_KEY = ""

# Dict with the parameters for CAMS European air quality analysis.
# More info here: https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-europe-air-quality-forecasts?tab=overview

PARAMS = {
    "model": "ensemble",  # model name
    "time": [  # hours of the day to retrieve measurements for
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00",
    ],
    "level": "0",  # Level 0 stands for 0 meters above ground, or ground level.
    "leadtime_hour": "0", # Any other vallue than 0 would fetch the forecasts
    "type": "analysis",  # there are two types available, analysis and forecast. Forecast is of no use for now.
}

AREAS = {  # extent of the area of interest for which the data should be selected
    "ireland": [
        55.65, -11.35, 51.35,
        -5.25,
    ],
}


def _to_datetime(dataset):
    """Convert the time column of newly downloaded CAMS analysis data into datetime objects"""

    # Strip the start date from the ANALYSIS attribute
    start_date = datetime.datetime.strptime(dataset.ANALYSIS[:16], "Europe, %Y%m%d")

    def convert_to_datetime(x):  # inner function used to add the timedelta to the start date
        return start_date + x

    dataset["time"] = dataset.time.to_pandas().apply(convert_to_datetime)  # convert the time parameter to datetime

    return dataset


def _calc_time_difference(start_date):
    """The ADS api allows up to 5000 fields per request. This function calculates the no. of hours between the
    current date minus two days and the last date for which data is present in the CAMS dataset."""

    end_date = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=2)
    difference = (end_date - start_date).days * 24  # difference between start date and end date in hours

    return difference


def _calc_download_periods(last_date, hour_interval=4976):
    """
    The ADS api allows up to 5000 fields per request. In case the difference in hours is over 5000 the download must
    be split into multiple requests. This function generates a dictionary containing the start and end date for each
    download, while making sure that the number of fields for the request do not exceed 5000 hours.

    :param last_date: datetime obj. containing the last date for which data is available in the downloaded CAMS product
    :param hour_interval: the number of hours/fields to allow for each request.
    :return: dictionary containing the start and end dates for each api download call
    """

    # Calculate the time difference in hours
    time_difference = _calc_time_difference(start_date=last_date)
    # Calculate how many download periods are needed
    no_downloads = int(np.ceil(time_difference/hour_interval))

    download_periods = {}  # empty dict to store the dates for each download

    if no_downloads == 1:  # In case only a single download is needed

        # Set the end data to today minus two days - in case the data for the last two days is not available yet
        end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=2)

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

        if not (d == no_downloads):  # if not the last download

            if d == 1:  # first download starting from the last available date for the existing dataset
                start = last_date
                end = start + timedelta(hours=hour_interval)
            else:  # not the first/last download, so we can use the information already in the download periods dict
                start = download_periods[f"download_{d-1}"]["end"] + timedelta(days=1)
                end = start + timedelta(hours=hour_interval)

        else:  # for the last download the final date will be less than the indicated hour interval
            start = download_periods[f"download_{d-1}"]["end"] + timedelta(days=1)
            end = datetime.datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=2)

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
    for pollutant in POLLUTANTS:

        # directory for files downloaded using the ADS api
        if not os.path.exists(f"{DATA_DIR_CAMS}/temp/api"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_CAMS}/temp/api", mode=0o777)

        # directory for files downloaded from the S3 bucket
        if not os.path.exists(f"{DATA_DIR_CAMS}/temp/s3"):  # Create the directory in case it does not exist
            os.makedirs(f"{DATA_DIR_CAMS}/temp/s3", mode=0o777)

        # Load the existing netcdf containing the CAMS data
        pollutant_name = POLLUTANTS[pollutant]["cams_var_name"]

        s3_client = boto3.client(  # initiate S3 client
            "s3",
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            endpoint_url=S3_ENDPOINT
        )

        s3_client.download_file(  # download the dataset currently stored in the S3 bucket
            S3_BUCKET_CAMS,
            f"{pollutant_name}.zip",
            f"{DATA_DIR_CAMS}/temp/s3/{pollutant_name}.zip"
        )

        # unzip the file downloaded from the s3 bucket
        with zipfile.ZipFile(f"{DATA_DIR_CAMS}/temp/s3/{pollutant_name}.zip", "r") as zip_ref:
            zip_ref.extractall(f"{DATA_DIR_CAMS}/temp/s3/")

        # load & copy dataset
        existing_pollutant_dataset = xr.load_dataset(f"{DATA_DIR_CAMS}/temp/s3/{pollutant_name}.nc")
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

            call = {  # set the API parameters
                "model": PARAMS["model"],
                "date": f"{start_date}/{end_date}",
                "format": "netcdf",
                "area": AREAS["ireland"],
                "time": PARAMS["time"],
                "variable": [pollutant],
                "level": PARAMS["level"],
                "leadtime_hour": PARAMS["leadtime_hour"],
                "type": PARAMS["type"],
            }

            # path to location where we want to store the download
            new_dataset_filename = f"{DATA_DIR_CAMS}/temp/api/{pollutant_name}_{start_date}_{end_date}.nc"

            if not os.path.exists(new_dataset_filename):
                try:
                    c = cdsapi.Client(
                        url=ADS_URL,
                        key=ADS_KEY
                    )  # Initiate the API client

                    c.retrieve(  # make the api call
                        "cams-europe-air-quality-forecasts",
                        call,
                        new_dataset_filename
                    )
                except Exception as e:
                    print(  # in case of an error
                        f"The API request for {pollutant} over the period {start_date} to {end_date}"
                        f" gave the following error:\n{e}")

        new_data = [  # open all the newly downloaded datasets and convert the time column to datetime
            _to_datetime(xr.open_dataset(f)) for f in glob.glob(f"{DATA_DIR_CAMS}/temp/api/{pollutant_name}*.nc")
        ]

        # create list of all the datasets to combine into the new dataset
        all_data = [existing_pollutant_dataset_cp] + new_data

        # combine the datasets into single dataset
        updated_dataset = xr.combine_nested(all_data, concat_dim="time")

        # Write the backup file & updated file as netcdf files
        existing_pollutant_dataset_cp.to_netcdf(f"{DATA_DIR_CAMS}/temp/{pollutant_name}_backup.nc")
        updated_dataset.to_netcdf(f"{DATA_DIR_CAMS}/temp/{pollutant_name}.nc")

        # zip the new nc file and the backup nc file
        _zip_nc_file(f"{DATA_DIR_CAMS}/temp/", f"{pollutant_name}_backup", f"{pollutant_name}_backup")
        _zip_nc_file(f"{DATA_DIR_CAMS}/temp/", pollutant_name, pollutant_name)

        # upload the compressed folders to s3 buckets, overwriting the existing objects
        s3_client.upload_file(
            f"{DATA_DIR_CAMS}/temp/{pollutant_name}_backup.zip",
            S3_BUCKET_CAMS,
            f"{pollutant_name}_backup.zip"
        )

        s3_client.upload_file(
            f"{DATA_DIR_CAMS}/temp/{pollutant_name}.zip",
            S3_BUCKET_CAMS,
            f"{pollutant_name}.zip"
        )

        # Finally, remove all the files from the temp folder and clear the memory
        existing_pollutant_dataset_cp = None
        updated_dataset = None
        new_data = None
        all_data = None

        rmtree(f"{DATA_DIR_CAMS}/temp/")


if __name__ == "__main__":
    main()
