# -*- coding: utf-8 -*-
# Author: Emma Chalençon
# Creation date: 27/01/2022
# Python version 3.8.5

import os
from os import path
import numpy as np
from osgeo import gdal, osr

def ArraytoRaster(array, outputRaster):
    """
    Function that writes an array to a raster file

    """
    if not path.exists(outputRaster):
        ncols, nrows = np.shape(array)
        driver = gdal.GetDriverByName('GTiff')
        outputRaster= driver.Create(outputRaster,nrows,ncols,1,gdal.GDT_Float64)
        outputRaster.SetGeoTransform(geotransform)
        outband = outputRaster.GetRasterBand(1)
        outband.WriteArray(array)                    
        outputRaster.SetProjection(srs.ExportToWkt())
        outputRaster.FlushCache()
        outputRaster = None
    return outputRaster

tiles = ["A01","A02","A03","A04","A09","B01","B02","B03","B04","B05","B07",
          "B08","B09","B10","B11","C01","C02","C03","C04","C05","C06","C07",
          "C08","C09","C10","C11","D01","D02","D03","D04","D05","D06","D07",
          "D08","D09","D10","D11","E01","E02","E03","E04","E05","E06","E07",
          "E08","E09","E10","E11","E12","E13","E14","F01","F02","F03","F04",
          "F05","F06","F07","F08","F09","F10","F11","F12","F13","F14","F15",
          "G02","G03","G04","G05","G06","G07","G08","G09","G10","G11","G12",
          "G13","G14","G15","H03","H04","H05","H06","H07","H08","H09","H10",
          "H11","H13","H14","H15","I03","I04","I05","I06","I07","I08","I09",
          "I10","I11","I12","I14","I15","J03","J04","J05","J06","J07","J08",
          "J09","J10","J11","K03","K04","K05","K06","K07","K08","K09","K10",
          "K11"]


for tile in tiles: 
    # Parameters
    workplace = "D:/New folder/Workplace/CloudFree/L8/" #output directory
    if not os.path.isdir(workplace):
        os.makedirs(workplace)
    imgList = os.listdir(workplace)
    srs = osr.SpatialReference()                 
    srs.ImportFromEPSG(2157) # chosen coordinate reference system 
    
    # Read imagery files 
    
    infilePath = "E:/Emma/FLARES/Image_processing/Landsat8_imagery/"+ tile +"/" #imagery directory
    fileList = os.listdir(infilePath)
    for file in fileList: 
        if file.endswith(".dat"):
            print("----------------------------------------")
            date = file [-16:-8]
            print("Date: " + date)
            
            tile = file [-7:-4]
            print("Tile: " + tile)
            
            CLOUDFree = workplace + tile + "_" + date + "_CloudFree.tif" 
            if not path.exists(CLOUDFree):
                print("\n Let's start the delineation process:")
 
                if not path.exists(CLOUDFree):
                    ds = gdal.Open(infilePath+file)        
                    dataset = gdal.Open(infilePath+file)
                    geotransform = dataset.GetGeoTransform()
                    
                    blue = ds.GetRasterBand(2).ReadAsArray()
                    green = ds.GetRasterBand(3).ReadAsArray()
                    nir = ds.GetRasterBand(5).ReadAsArray()
                    swir = ds.GetRasterBand(6).ReadAsArray()
                    
                    blue = blue.astype('f4')
                    green = green.astype('f4')
                    nir = nir.astype('f4')
                    swir = swir.astype('f4')
                    np.seterr(divide='ignore', invalid='ignore')

                    swm = (blue+green)/(nir+swir) #cloud and water mask
                    swm[swm>=0.4]=np.nan
                    swm[swm<0.4]=1
                                
                    ArraytoRaster(swm, CLOUDFree)
                
                print("___ for " + date + ": Cloudfree raster created and filtered with SWM NDVI and RGB")



years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
for tile in tiles:
    for year in years: 
        array_list = []
        for file in os.listdir(workplace):
            if file.startswith(tile +'_'+ year) and file.endswith(".tif"):
                img_ds = gdal.Open(workplace + file).ReadAsArray()
                dataset = gdal.Open(workplace + file)
                geotransform = dataset.GetGeoTransform()
                img_ds[img_ds==np.nan]=0
                array_list.append(img_ds)       
        
        if len(array_list) >= 1 :  
            array_out = np.nansum(array_list, axis=0)
            finalArray = array_out 
            finalArray[finalArray<=0]=np.nan
            addRaster = workplace + tile + "_" + year + "_CF_ADD.tif"
            if not path.exists(addRaster):
                ArraytoRaster(finalArray, addRaster)
            print("______ ", len(array_list), "results combined")
