# -*- coding: utf-8 -*-

# Author: Emma Chalençon
# Creation date: 27/08/2021
# Python version 3.8.5


import fiona
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import pandas as pd
import os
from os import path
import numpy as np
from osgeo import gdal, ogr, osr

def reprojection(raster):
    """
    Function that reprojects a raster in the crs given by the variable dst_crs

    """
    proj_imagery = workplace + tile + "_" + date + "_" + bandN +'.tif'
    if not path.exists(proj_imagery):
        transform, width, height = calculate_default_transform(
            raster.crs, dst_crs, raster.width, raster.height, *raster.bounds)
        kwargs = raster.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height})
    
        with rasterio.open(proj_imagery, 'w', **kwargs) as dst:
            for i in range(1, raster.count + 1):
                reproject(
                    source=rasterio.band(raster, i),
                    destination=rasterio.band(dst, i),
                    raster_transform=raster.transform,
                    raster_crs=raster.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    new_image=rasterio.open(proj_imagery)
    return new_image

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
    
def Polygonize (raster, shapefile):
    """
    Function that creates a shapefile from a raster file

    """
    if not path.exists(shapefile):
        src_ds = gdal.Open(raster)
        srcband = src_ds.GetRasterBand(1)
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(shapefile)
        dst_layer = dst_ds.CreateLayer(shapefile, srs)
        newField = ogr.FieldDefn('DN', ogr.OFTReal)
        dst_layer.CreateField(newField)
        gdal.Polygonize(srcband, srcband, dst_layer, 0, [])
        dst_ds.Destroy()
        src_ds = None
    return shapefile 

def ClipRwithS (shapefile, raster, outraster):
    """
    Function that clips a raster with a shapefile

    """
    if not path.exists(outraster):
        with fiona.open(shapefile, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform})
        with rasterio.open(outraster, "w", **out_meta) as dest:
            dest.write(out_image)

def ClipSwithS (Clipped, Clipper, New):
    """
    Function that clips a shapefile with another shapefile

    """
    if not path.exists(New):
        Clipper = gpd.read_file(Clipper)
        Clipped = gpd.read_file(Clipped)
        clipped = gpd.clip(Clipped,Clipper)
        clipped.to_file(New)
    return New
    
def FilterPolyArea (shapefile,areaN):
    """
    Function that delete every feature of a shapefile whose area is below the threshold areaN (m2)

    """ 
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 1)
    layer = dataSource.GetLayer()
    schema = []
    ldefn = layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)
    if "Area" not in schema: 
        new_field = ogr.FieldDefn("Area", ogr.OFTReal)
        new_field.SetWidth(32)
        new_field.SetPrecision(3)
        layer.CreateField(new_field)
    
    i = 0
    for feature in layer :
        i += 1
        geom = feature.GetGeometryRef()
        area = geom.GetArea() 
        if area<=areaN:
            layer.DeleteFeature(feature.GetFID())
            continue
        feature.SetField("Area", area)
        layer.SetFeature(feature)
    
    j = 0    
    for feature in layer :
        j+= 1
     
    print("______ Features before =", i)
    print("______ Features after =", j)

    dataSource = None

def boundingBoxToOffsets(bbox, geot):
    """
    Function that retrieves offsets 

    """
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]
 
def geotFromOffsets(row_offset, col_offset, geot):
    """
    Function that retrieves geotransform parameters 

    """
    new_geot = [geot[0] + (col_offset * geot[1]), geot[1], 0.0, geot[3] + (row_offset * geot[5]), 0.0, geot[5]]
    return new_geot

def setFeatureStats(fid, max, names=["max","fid"]):
    """
    Function that calculates stats per feature 

    """
    featstats = {names[0]: max, names[1]: fid,}
    return featstats

def seedintersection (fn_raster, fn_zones, elim):
    """
    Function that deletes every feature which doesn't contain a high-intensity seed 
    
    fn_raster : raster containing the values per pixel
    fn_zones : polygon shapefiles 
    elim : high-intensity threshold 
    
    if within the extend of a polygon in fn_zones, the pixel with the highest value 
    in fn_raster is inferior to elim then the feature will be deleted

    """
    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"
    r_ds = gdal.Open(fn_raster)
    p_ds = ogr.Open(fn_zones, 1) 
    lyr = p_ds.GetLayer()
    geot = r_ds.GetGeoTransform()
    nodata = r_ds.GetRasterBand(1).GetNoDataValue()     
    zstats = [] 
    p_feat = lyr.GetNextFeature()

    while p_feat:
        if p_feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                mem_driver.DeleteDataSource(shp_name)
            
            tp_ds = mem_driver.CreateDataSource(shp_name)
            tp_lyr = tp_ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
            tp_lyr.CreateFeature(p_feat.Clone())
            offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
            geot)
            new_geot = geotFromOffsets(offsets[0], offsets[2], geot)
             
            tr_ds = mem_driver_gdal.Create(\
            "", \
            offsets[3] - offsets[2], \
            offsets[1] - offsets[0], \
            1, \
            gdal.GDT_Byte)
             
            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
            tr_array = tr_ds.ReadAsArray()
             
            r_array = r_ds.GetRasterBand(1).ReadAsArray(\
            offsets[2],\
            offsets[0],\
            offsets[3] - offsets[2],\
            offsets[1] - offsets[0])
             
            id = p_feat.GetFID()
            
            r_array[np.isnan(r_array)]=0
            
            if r_array is not None:
                maskarray = np.ma.MaskedArray(\
                r_array,\
                mask=np.logical_or(r_array==nodata, np.logical_not(tr_array)))
                 
                if maskarray is not None:
                    zstats.append(setFeatureStats(id, maskarray.max()))
                else:
                    zstats.append(setFeatureStats(id, nodata))
            else:
                zstats.append(setFeatureStats(id, nodata))
   
            tp_ds = None
            tp_lyr = None
            tr_ds = None
             
            p_feat = lyr.GetNextFeature()
    
    deleteList = []
    for item in zstats:
        for key, value in item.items():
            if key == "max" and value <= elim:
                    deleteList.append(item["fid"])
    i = 0
    for feature in lyr :
        i+=1
        if feature.GetFID() in deleteList:
            lyr.DeleteFeature(feature.GetFID())
    j = 0    
    for feature in lyr :
        j+= 1
     
    print("______ Features before =", i)
    print("______ Features after =", j)
    r_ds = None
    p_ds = None
    
def timestamp(date_raster, yearly_shapefile): 
    """
    Function that timestamps the final burnscar yearly features.
    
    yearly_shapefile : shapefile with the yearly burnscars features
    date_raster : raster with burnscar delineation results for a given date
    
    If a feature in yearly_shapefile corresponds to a result on this given date's raster 
    then the feature will be timestamped (as an attribute) with this given date.
    
    """
    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"
    r_ds = gdal.Open(date_raster)
    p_ds = ogr.Open(yearly_shapefile, 1) 
    lyr = p_ds.GetLayer()
    geot = r_ds.GetGeoTransform()
    nodata = r_ds.GetRasterBand(1).GetNoDataValue()     
    zstats = [] 
    p_feat = lyr.GetNextFeature()
    
    while p_feat:
        if p_feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                mem_driver.DeleteDataSource(shp_name)
            
            tp_ds = mem_driver.CreateDataSource(shp_name)
            tp_lyr = tp_ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
            tp_lyr.CreateFeature(p_feat.Clone())
            offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
            geot)
            new_geot = geotFromOffsets(offsets[0], offsets[2], geot)
             
            tr_ds = mem_driver_gdal.Create(\
            "", \
            offsets[3] - offsets[2], \
            offsets[1] - offsets[0], \
            1, \
            gdal.GDT_Byte)
             
            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
            tr_array = tr_ds.ReadAsArray()
             
            r_array = r_ds.GetRasterBand(1).ReadAsArray(\
            offsets[2],\
            offsets[0],\
            offsets[3] - offsets[2],\
            offsets[1] - offsets[0])
             
            id = p_feat.GetFID()
            r_array[pd.isnull(r_array)]=0
            
            if r_array is not None:
                maskarray = np.ma.MaskedArray(\
                r_array,\
                mask=np.logical_or(r_array==nodata, np.logical_not(tr_array)))
                 
                if maskarray is not None:
                    zstats.append(setFeatureStats(id, maskarray.max()))
                else:
                    zstats.append(setFeatureStats(id, nodata))
            else:
                zstats.append(setFeatureStats(id, nodata))
       
            tp_ds = None
            tp_lyr = None
            tr_ds = None
             
            p_feat = lyr.GetNextFeature()
    
    stampList = []
    for item in zstats:
        for key, value in item.items():
            if key == "max" and value == 2:
                    stampList.append(item["fid"])

    ldefn = lyr.GetLayerDefn()
    schema = []
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)
    if "Date" not in schema: 
        new_field = ogr.FieldDefn("Date", ogr.OFTString)
        lyr.CreateField(new_field)
    
    for feature in lyr :
        if feature.GetFID() in stampList:
            if feature.GetField("Date") is None: 
                datestamp = fn_raster[-18:-10]
                feature.SetField("Date", datestamp)
                lyr.SetFeature(feature)
            
    r_ds = None
    p_ds = None

def ClipRwithR(extent_raster, original_raster, clipped_raster):
    """
    Function that clips a raster with another raster's extent

    """
    if not path.exists(clipped_raster):
        data = gdal.Open(extent_raster)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        gdal.Translate(clipped_raster, original_raster, projWin = [minx, maxy, maxx, miny])

# Parameters
workplace = "D:/Workplace/" #workplace directory
imgList = os.listdir(workplace) 
srs = osr.SpatialReference()                 
srs.ImportFromEPSG(2157)
dst_crs = 'EPSG:2157' # chosen coordinate reference system 

# Read imagery files 
infilePath = "E:/Sentinel2_imagery/" #imagery directory
fileList = os.listdir(infilePath)

for file in fileList:
    print("----------------------------------------")
    date = file [11:19]
    print("Date: " + date)
    tile = file[39:44]
    print("Tile: " + tile)
    AOI = "E:/S2 tiles/" + tile + ".shp" #path to S2 tiles shapefiles
    shape_file = gpd.read_file(AOI)
    
    subfolder = infilePath+"\\"+file+"\\"+"GRANULE"
    subsubfolder = os.listdir(subfolder)
    infileFolder10 = subfolder+"\\"+subsubfolder[0]+"\\"+"IMG_DATA\R10m"
    infileFolder20 = subfolder+"\\"+subsubfolder[0]+"\\"+"IMG_DATA\R20m"
    finalraster = workplace + tile + "_" + date + "_final.tif"
    if not path.exists(finalraster):
        bandsList = os.listdir(infileFolder10)
        for band in bandsList:
            if band[23]=="B":
                source_imagery = rasterio.open(infileFolder10+"\\"+band)
                gt = source_imagery.transform
                xRes = gt[0]
                yRes = -gt[4]
                bandN = band[23:26]
                print('___ Imagery file ' + bandN +' Projection: ', source_imagery.crs)
                
                # Reproject imagery file 
                new_image = reprojection(source_imagery)
                print("______ " + bandN + ' new projection: ', new_image.crs)
                
                # Resample imagery file
                input_file = workplace + tile + "_" + date + "_" + bandN +".tif"
                resampled = workplace + tile + "_" + date + "_" + bandN + "_S.tif"
                if not path.exists(resampled):
                    ds = gdal.Translate(resampled, input_file, xRes=xRes, yRes=xRes, resampleAlg="bilinear")
                    print("______ " + bandN + ' resampled')
                
                # Clip imagery file with AOI shapefile
                clipped = workplace + tile + "_" + date + "_" + bandN +'_clip.tif'
                ClipRwithS(AOI, resampled, clipped)
                print("______ " + bandN + ' clipped')
        
        bandsList = os.listdir(infileFolder20)
        for band in bandsList:
            if band[23:26]=="B11" or band[23:26]=="B12" :
                source_imagery = rasterio.open(infileFolder20+"\\"+band)
                bandN = band[23:26]
                print('___ Imagery file ' + bandN +' Projection: ', source_imagery.crs)
                
                # Reproject imagery file 
                new_image = reprojection(source_imagery)
                print("______ " + bandN + ' new projection: ', new_image.crs)
                
                # Resample imagery file
                input_file = workplace + tile + "_" + date + "_" + bandN +".tif"
                resampled = workplace + tile + "_" + date + "_" + bandN + "_S.tif"
                if not path.exists(resampled):
                    ds = gdal.Translate(resampled, input_file, xRes=xRes, yRes=yRes, resampleAlg="bilinear")
                    print("______ " + bandN + ' resampled')
                
                # Clip imagery file with original 10m raster extent
                clipped = workplace + tile + "_" + date + "_" + bandN +'_clip.tif'
                extent = workplace + tile + "_" + date + "_B03_clip.tif"
                ClipRwithR(extent, resampled, clipped)
                print("______ " + bandN + ' clipped')
        
        print("\n => " + date + ": Imagery ready") 
        print("\n Let's start the delineation process:") 
        
        imgList = os.listdir(workplace)
        NDVI = workplace + tile + "_" + date + "_ndvi.tif"
        NBRFILTER = workplace + tile + "_" + date + "_nbrfilter.tif"  
        if not path.exists(NDVI): 
            for img in imgList:
                if img.endswith(tile + "_" + date+"_B02_clip.tif"):
                    blue = gdal.Open(workplace + img).ReadAsArray()
                    dataset = gdal.Open(workplace + img)
                    geotransform = dataset.GetGeoTransform()
                if img.endswith(tile + "_" + date+"_B03_clip.tif"):
                    green = gdal.Open(workplace +img).ReadAsArray()
                if img.endswith(tile + "_" + date+"_B04_clip.tif"):
                    red = gdal.Open(workplace + img).ReadAsArray()
                if img.endswith(tile + "_" + date+"_B08_clip.tif"):
                    nir = gdal.Open(workplace + img).ReadAsArray()
                if img.endswith(tile + "_" + date+"_B11_clip.tif"):
                    swir = gdal.Open(workplace + img).ReadAsArray()
                if img.endswith(tile + "_" + date+"_B12_clip.tif"):
                    swir2 = gdal.Open(workplace + img).ReadAsArray()
            
            swir = swir.astype('f4')
            blue = blue.astype('f4')
            green = green.astype('f4')
            red = red.astype('f4')
            nir = nir.astype('f4')
            swir2 = swir2.astype('f4')
            np.seterr(divide='ignore', invalid='ignore')
            
            white = blue + green + red
            white[white>=2000]= np.nan
            white[white<2000]= 1
            
            swm = (blue+green)/(nir+swir)   #cloud and water mask
            swm[swm>=0.4]=np.nan
            swm[swm<0.4]=1
            
            ndvi = (nir-red)/(nir+red)
            ndvi_export = ndvi*swm
            ArraytoRaster(ndvi_export, NDVI)
            
            ndvi[ndvi>=0.6]= np.nan
            ndvi[ndvi<=0.25]= np.nan
            ndvi[ndvi<=0.6]= 1
      
            nbr = (nir - swir2) / (nir + swir2)
            nbr[nbr >= 0.01]=np.nan
            nbr[nbr <= -0.05]=2
            nbr[nbr < 0.01]=1
            
            nbrfilter = nbr*swm*white*ndvi
            
            ArraytoRaster(nbrfilter, NBRFILTER)
         
        print("___ for " + date + ": NBR raster created and filtered with SWM NDVI and RGB")
        
        ## Mask the results with CLC2018 PRIME2 WFD and home-made constant false alarm mask
        
        Local_mask = "E:/Masks/" + tile + "_mask.shp" #path the S2 masks 
        NBR_masked =  workplace + tile + "_" + date + "_NBR_masked.tif"
        ClipRwithS(Local_mask, NBRFILTER, NBR_masked)
        
        ## Select only areas over 0.4ha 
        
        resultsRaster= workplace + tile + "_" + date + "_results.tif"
        if not path.exists(resultsRaster): 
            res = gdal.Open(NBR_masked).ReadAsArray()
            dataset = gdal.Open(NBR_masked)
            geotransform = dataset.GetGeoTransform()
            res = res.astype('f4')
            res[res==1]=2
            ArraytoRaster(res, resultsRaster)
            dataset = None 
        resultsPoly = resultsRaster[:-4] + "_poly.shp" 
        if not path.exists(resultsPoly):
            outPoly = Polygonize (resultsRaster, resultsPoly)
        print("___ for " + date + ": Areas under 0.4 ha filtered")    
        FilterPolyArea (resultsPoly,4000)
        
        ## Only keep polygons which contain a NBR_masked high intensity seed (<= -0.05)
        
        print("___ for " + date + ": Polygons without high intensity seeds discarded")
        seedintersection (NBR_masked, resultsPoly, 1.0)
        finalraster = workplace + tile + "_" + date + "_final.tif"
        if not path.exists(finalraster):
            ras_ds = gdal.Open(NBR_masked) 
            vec_ds = ogr.Open(resultsPoly) 
            lyr = vec_ds.GetLayer() 
            geot = ras_ds.GetGeoTransform() 
            drv_tiff = gdal.GetDriverByName("GTiff") 
            chn_ras_ds = drv_tiff.Create(finalraster, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
            chn_ras_ds.SetGeoTransform(geot)
            chn_ras_ds.SetProjection(srs.ExportToWkt())
            gdal.RasterizeLayer(chn_ras_ds, [1], lyr, options=['ATTRIBUTE=DN']) 
            chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0) 
            chn_ras_ds.FlushCache()
            chn_ras_ds = None
    
        print("\n => " + date + ": burn area delineation done \n") 

print("----------------------------------------")
print("\n FINAL STEP")

imgList = os.listdir(workplace)
years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
months = ["01","02","03","04","05","06"]
tiles = ["29UMT","28UGC","28UGD","29ULT","29UMA","29UMS","29UMT","29UMU","29UMV",
          "29UNA","29UNB","29UNT","29UNU","29UNV","29UPA","29UPB","29UPT","29UPU",
          "29UPV","29UQU","30UUE"]
for tile in tiles:
    print("\n" + "----------------------------------------")
    print(tile)
    print("----------------------------------------")    
    for year in years:
        img_list=[]
        array_list=[]
        min_list=[]
        max_list=[]
        
        for img in imgList:
            
        ## Add all the burnscar polygons from the same tile/year:
            
            if img.endswith("_final.tif") and img.startswith(tile+"_"+year):
                img_list.append(img)
                img_ds = gdal.Open(workplace + img).ReadAsArray()
                dataset = gdal.Open(workplace + img)
                geotransform = dataset.GetGeoTransform()
                img_ds[img_ds==np.nan]=0
                array_list.append(img_ds)
        
        ## Calculate the difference between the min ndvi Jan-Jui/ the max ndvi Aug-Dec:
            # if burn scar: positive dif of at least 0.1 but not higher than 0.6
            
            if img.endswith("_ndvi.tif") and img.startswith(tile+"_"+year):
                n = len(tile)+5
                print(img[n:n+2])
                if img[n:n+2] in months:
                    print(img, "in months")
                    img_ds = gdal.Open(workplace + img).ReadAsArray()
                    dataset = gdal.Open(workplace + img)
                    geotransform = dataset.GetGeoTransform()
                    min_list.append(img_ds)
                if img[n:n+2] not in months:
                    print(img, " not in months")
                    img_list.append(img)
                    img_ds = gdal.Open(workplace + img).ReadAsArray()
                    dataset = gdal.Open(workplace + img)
                    geotransform = dataset.GetGeoTransform()
                    max_list.append(img_ds)
                    
        if len(min_list) >= 1 :
            addin = np.stack(min_list, axis=0 )
            Min = np.nanmin(addin, axis=0)      
            if len(max_list) >= 1 :
                addax = np.stack(max_list, axis=0 )
                Max = np.nanmax(addax, axis=0)    
            
                Diff = Max-Min  
                MAX = workplace + tile + "_" + year + "_Max.tif"
                MIN = workplace + tile + "_" + year + "_Min.tif"
                if not path.exists(MIN):
                    ArraytoRaster(Min, MIN)
                    ArraytoRaster(Max, MAX)
                Diff[Diff>=0.6]= np.nan
                Diff[Diff<=0.1]= np.nan
                Diff[Diff<=0.6]= 1
                Diff[Diff==np.nan]=0
                DIFF = workplace + tile + "_" + year + "_Diff.tif"
                if not path.exists(DIFF):
                    ArraytoRaster(Diff, DIFF)    
        print("\n In", year, len(array_list), "images analysed")            
        
        if len(array_list) >= 1 :  
            array_out = sum(array_list)
            finalArray = array_out * Diff
            finalArray[finalArray<=0]=np.nan
            addRaster = workplace + tile + "_" + year + "_ADD.tif"
            if not path.exists(addRaster):
                ArraytoRaster(finalArray, addRaster)
            print("___", len(array_list), "results combined")
            
            finalArray[finalArray>=2]=2        
            addRasterpoly = workplace + tile + "_" + year + "_ADDpoly.tif"
            if not path.exists(addRasterpoly):
                ArraytoRaster(finalArray, addRasterpoly)
            finalPoly = workplace + tile + "_" + year + ".shp"
            if not path.exists(finalPoly):
                Polygonize (addRasterpoly, finalPoly)
            print("___ and polygonized")
            
            print("___ Polygons present on one single date only discarded")
            seedintersection (addRaster, finalPoly, 3.0)
            FilterPolyArea (finalPoly,4000)
        
            img_list.sort()
            for img in img_list:
                timestamp(workplace+img, finalPoly)
            print("___ Polygons time-stamped")   
         
    print("\n----------------------------------------")
