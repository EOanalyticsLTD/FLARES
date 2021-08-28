"""
Quick program to set up the data folder structure and download the data for WP4 from the S3 buckets.
"""

import os
import glob
import pytz
import shutil
import zipfile
import pathlib
import datetime
import argparse
from pathlib import Path

import boto3
from botocore.config import Config

from constants import DATA_DIR, DATA_DIR_CAMS, DATA_DIR_ERA5, DATA_DIR_MERA, DATA_DIR_GFAS, DATA_DIR_LC, DATA_DIR_PLOTS, \
    DATA_DIR_DEM, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION_NAME, DATA_DIR_FIRES

# Dictionary containing the directory location for each of the datasets as a key, storing the name of bucket and the
# file names of the objects to download as values in the dict.

DATASETS = {
    DATA_DIR_CAMS:
        {
            'bucket':'cams-analysis',
            'filenames':[
                'co_conc',
                'no2_conc',
                'no_conc',
                'o3_conc',
                'pm10_conc',
                'pm2p5_conc',
                'so2_conc',
                'pmwf_conc',
                'fire_mask',
            ],
        },
    DATA_DIR_ERA5:
        {
            'bucket':'era5-reanalysis',
            'filenames':[
                'era5',
            ],
        },
    DATA_DIR_MERA:
        {
            'bucket':'mera-reanalysis',
            'filenames':[
                'mera',
            ]
        },
    DATA_DIR_GFAS:
        {
            'bucket':'gfas',
            'filenames':[
                'frp',
                'wildfire_flux',
            ],
        },
    DATA_DIR_LC:
        {
            'bucket':'wp4-land-cover',
            'filenames':[
                'corine_ireland',
            ],
        },
    DATA_DIR_DEM:
        {
            'bucket':'wp4-dem-slope',
            'filenames':[
                'dem_ireland',
                'slope_ireland',
            ],
        }
}

# Configuration for the S3 client
BOTO_CONFIG = Config(
        retries=dict(
            max_attempts=5
        )
    )


def set_up_folder_structure():
    """
    Function that sets up the projects folder structure
    """
    if not os.path.exists(DATA_DIR):
        print(f'Base directory "{DATA_DIR}" does not exist. Creating base directory.')
        os.makedirs(DATA_DIR)

    for dir_path in [DATA_DIR_CAMS, DATA_DIR_ERA5, DATA_DIR_MERA, DATA_DIR_GFAS,
                     DATA_DIR_LC, DATA_DIR_DEM, DATA_DIR_PLOTS, DATA_DIR_FIRES]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def download_data():
    """
    Function for downloading the data relevant to the project from the S3 buckets
    """

    # for each directory
    for dir_string in [DATA_DIR_CAMS, DATA_DIR_MERA, DATA_DIR_ERA5, DATA_DIR_GFAS, DATA_DIR_LC, DATA_DIR_DEM]:

        s3_files = DATASETS[dir_string]['filenames']
        s3_bucket = DATASETS[dir_string]['bucket']

        dir_path = pathlib.Path(dir_string)  # convert to path

        if not os.path.exists(dir_path.joinpath('temp')):  # create a temp dir to store the download
            os.makedirs(dir_path.joinpath('temp'))

        for file in s3_files:

            # in case the file is already present, download is skipped
            if len(glob.glob(f'{dir_path.joinpath(file).as_posix()}*')) > 0:
                print(f'File {file} already present in directory')
                continue

            print(f'Starting download from s3 bucket for "{file}.zip"')
            try:
                # Initiatie the S3 client
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=S3_ACCESS_KEY,
                    aws_secret_access_key=S3_SECRET_KEY,
                    endpoint_url=S3_ENDPOINT,
                    region_name=S3_REGION_NAME,
                    config=BOTO_CONFIG,
                )

                s3_client.download_file(
                    s3_bucket,
                    f'{file}.zip',
                    dir_path.joinpath(f'temp/{file}.zip').as_posix()
                )

            except Exception:
                raise
            else:
                print(f'Download for "{file}.zip" completed')

            with zipfile.ZipFile(dir_path.joinpath(f'temp/{file}.zip'), 'r') as zip_ref:
                zip_ref.extractall(dir_path)

        shutil.rmtree(dir_path.joinpath('temp/'))


def update_data():
    """
    Function to check for any updates in the datasets
    """

    for dir_path in [DATA_DIR_CAMS, DATA_DIR_ERA5, DATA_DIR_GFAS]:

        s3_files = DATASETS[dir_path]['filenames']
        s3_bucket = DATASETS[dir_path]['bucket']

        dir_path = pathlib.Path(dir_path)

        if not os.path.exists(dir_path.joinpath('temp')):
            os.makedirs(dir_path.joinpath('temp'))

        for file in s3_files:

            # Check last change on disk
            mod_time = os.path.getmtime(Path(dir_path).joinpath(f'{file}.nc'))
            last_mod_file = datetime.datetime.fromtimestamp(mod_time)
            last_mod_file = pytz.utc.localize(last_mod_file)

            # Check last change in bucket
            last_mod_s3 = s3_client.get_object(
                Bucket=s3_bucket,
                Key=f'{file}.zip'
            )['LastModified']

            # if last change in bucket more recent than on disk -> start download.
            if last_mod_s3 > last_mod_file:
                print(f'Updated version available for "{file}.zip", starting download from s3 bucket')
                try:
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY,
                        endpoint_url=S3_ENDPOINT,
                        config=BOTO_CONFIG,
                        region_name=S3_REGION_NAME,
                    )

                    s3_client.download_file(
                        s3_bucket,
                        f'{file}.zip',
                        dir_path.joinpath(f'temp/{file}.zip').as_posix()
                    )
                except Exception:
                    raise
                else:
                    print(f'Download for "{file}.zip" completed')

                with zipfile.ZipFile(dir_path.joinpath(f'temp/{file}.zip'), 'r') as zip_ref:
                    zip_ref.extractall(dir_path)

        shutil.rmtree(dir_path.joinpath('temp/'))


parser = argparse.ArgumentParser(description='Quick program to set up the data folder structure and download the data '
                                             'for WP4 from the S3 buckets.')

parser.add_argument("-f", "--folder_structure", help="Set up folder structure", action='store_true')
parser.add_argument("-d", "--download_data", help="Download data from s3 buckets", action='store_true')
parser.add_argument("-u", "--update_data",
                    help="Checks for updates and downloads the data from s3 buckets if needed", action='store_true')

args = parser.parse_args()


def main(options):
    """
    Main function - processes the arguments
    """
    # if the folder structure flag is given/True set up the structure
    if options.folder_structure:
        set_up_folder_structure()

    # if the download data flag is given/True download the data from the s3 buckets
    if options.download_data:
        download_data()

    # if the update data flag is given/True check for updated to the datasets in the s3 buckets
    if options.update_data:
        update_data()


if __name__ == '__main__':
    main(args)



