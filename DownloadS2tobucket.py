import os 
from os import path
import shutil
from mundilib import MundiCollection, get_node
import boto3

access_key = 'AXKWGHC81YQL7QNOXC56'
secret_key = 'e9hSQnyOqtlaoZT5MNKpAI2k3homSbom4iSnA60x'

downloader = MundiCollection("Sentinel2").mundi_downloader(access_key, secret_key)

downloader.browse(date_from='2018-01-01T12:00:00', date_to='2018-07-31T23:00:00', other_fields={"DIAS:tileIdentifier": "29UPU", "DIAS:productLevel" :  "L2A"})
print(len(downloader.records.items()))
i = 0
for id_, record in downloader.records.items():
    if float(get_node(record, 'DIAS:cloudCoverPercentage').text) <70:
        target_folder = "A:/"
        print(id_)
        print(get_node(record, 'DIAS:cloudCoverPercentage').text)
        print(get_node(record, "DIAS:sensingStartDate").text)
        i+=1 
        print(i)
        if not path.exists(target_folder+id_):
              downloader.download_by_id(id_, target_folder)
              
        ## Write to bucket

        endpoint = 'https://obs.eu-de.otc.t-systems.com'
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, endpoint_url=endpoint)
        bucket = '29upu'

        fileList = os.listdir(target_folder)

        for file in fileList:
            print("----------------------------------------")
            date = file [11:19]
            print("Date: " + date)
            tile = file[39:44]
            print("Tile: " + tile)
            nb = file [-10:-6]

            subfolder = target_folder+"/"+file+"/"+"GRANULE"
            subsubfolder = os.listdir(subfolder)
            infileFolder10 = subfolder+"/"+subsubfolder[0]+"/"+"IMG_DATA/R10m/"
            infileFolder20 = subfolder+"/"+subsubfolder[0]+"/"+"IMG_DATA/R20m/"
            bandsList = os.listdir(infileFolder10)
            for band in bandsList:
                 if band[27]=="B":
                    key = infileFolder10 + band
                    filename = tile + '_' + date + '_' + nb + '/'+ band
                    s3_client.upload_file(key, bucket, filename)
            bandsList2 = os.listdir(infileFolder20)
            for band2 in bandsList2:
                if band2[27:30]=="B11":
                    key2 = infileFolder20 + band2
                    filename2 = tile + '_' + date + '_' + nb + '/'+ band2
                    s3_client.upload_file(key2, bucket, filename2)
                if band2[27:30]=="B12":
                    key2 = infileFolder20 + band2
                    filename2 = tile + '_' + date + '_' + nb + '/'+ band2
                    s3_client.upload_file(key2, bucket, filename2)
        
        shutil.rmtree(target_folder) 
