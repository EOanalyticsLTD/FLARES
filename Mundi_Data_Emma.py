# -*- coding: utf-8 -*-

# Author: Emma Chalençon
# Creation date: 06/07/2021
# Python version 3.8.5

import os 
from os import path
import shutil
from mundilib import MundiCollection, get_node

access_key = 'AXKWGHC81YQL7QNOXC56'
secret_key = 'e9hSQnyOqtlaoZT5MNKpAI2k3homSbom4iSnA60x'

downloader = MundiCollection("Sentinel2").mundi_downloader(access_key, secret_key)
target_folder = "D:/Emma/FLARES/Image_processing/Sentinel2_imagery/"

## Download one image in particular

record_id = "S2B_MSIL2A_20210701T113319_N0301_R080_T29UPV_20210701T132705"
print("Will download {"+ record_id+"} to {"+target_folder+"}")
if not path.exists(target_folder+record_id):
    downloader.download_by_id(record_id, target_folder)

## Download every image corresponding to specific parameters

downloader.browse(date_from='2020-03-31T12:00:00', date_to='2020-05-31T23:00:00', other_fields={"DIAS:tileIdentifier": "29UPU", "DIAS:productLevel" :  "L2A"})
print(len(downloader.records.items()))
i = 0
for id_, record in downloader.records.items():
    if float(get_node(record, 'DIAS:cloudCoverPercentage').text) <70:
        print(id_)
        print(get_node(record, 'DIAS:cloudCoverPercentage').text)
        print(get_node(record, "DIAS:sensingStartDate").text)
        i+=1 
        print(i)
        if not path.exists(target_folder+id_):
              downloader.download_by_id(id_, target_folder)

## Zip folders
print(os.listdir(target_folder))
for dir_name in os.listdir(target_folder):
    output_filename = target_folder + dir_name 
    shutil.make_archive(output_filename, 'zip', target_folder+dir_name)

## Write to bucket

import boto3

endpoint = 'https://obs.eu-de.otc.t-systems.com'
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, endpoint_url=endpoint)
bucket = 'burnscars'
for file in os.listdir(target_folder):
    if file.endswith(".zip"):
        filename = file
        key = target_folder+file
        s3_client.upload_file(key, bucket, filename)

## Download from bucket

endpoint = 'https://obs.eu-de.otc.t-systems.com'
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, endpoint_url=endpoint)
bucket = 'burnscars'
listing = s3_client.list_objects_v2(Bucket=bucket)
number = len(listing["Contents"])
print(number)
for i in range(number):
    print(i)
    file_key = listing["Contents"][i]['Key']
    print(file_key)
    s3_client.download_file(bucket, file_key, file_key)

