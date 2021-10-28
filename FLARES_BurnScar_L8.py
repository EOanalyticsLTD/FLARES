# -*- coding: utf-8 -*-

# Author: Emma Chalençon
# Creation date: 27/08/2021
# Python version 3.8.5

import fiona
import rasterio
import rasterio.mask
import os
from os import path
import numpy as np
from osgeo import gdal, ogr, osr

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

# Parameters
workplace = "D:/Workplace/" #workplace directory
imgList = os.listdir(workplace)
srs = osr.SpatialReference()                 
srs.ImportFromEPSG(2157) # chosen coordinate reference system 

# Read imagery files 

infilePath = "C:/Landsat_imagery/" #imagery directory
fileList = os.listdir(infilePath)
for file in fileList: 
    if file.endswith(".dat"):
        print("----------------------------------------")
        date = file [-16:-8]
        print("Date: " + date)
        
        tile = file [-7:-4]
        print("Tile: " + tile)
        
        print("\n Let's start the delineation process:")
        
        NDVI = workplace + tile + "_" + date + "_ndvi.tif"
        NBRFILTER = workplace + tile + "_" + date + "_nbrfilter.tif" 
        
        if not path.exists(NDVI):
            ds = gdal.Open(infilePath+file)        
            dataset = gdal.Open(infilePath+file)
            geotransform = dataset.GetGeoTransform()
            
            blue = ds.GetRasterBand(2).ReadAsArray()
            green = ds.GetRasterBand(3).ReadAsArray()
            red = ds.GetRasterBand(4).ReadAsArray()
            nir = ds.GetRasterBand(5).ReadAsArray()
            swir = ds.GetRasterBand(6).ReadAsArray()
            swir2 = ds.GetRasterBand(7).ReadAsArray()
            
            blue = blue.astype('f4')
            green = green.astype('f4')
            red = red.astype('f4')
            nir = nir.astype('f4')
            swir = swir.astype('f4')
            swir2 = swir2.astype('f4')
            np.seterr(divide='ignore', invalid='ignore')
                    
            white = blue + green + red
            white[white>=1700]= np.nan
            white[white<1700]= 1
             
            swm = (blue+green)/(nir+swir) #cloud and water mask
            swm[swm>=0.4]=np.nan
            swm[swm<0.4]=1
            
            ndvi = (nir-red)/(nir+red)
            ndvi_export = ndvi*swm
            ArraytoRaster(ndvi_export, NDVI)
            
            ndvi[ndvi>=0.6]= np.nan
            ndvi[ndvi<=0.25]= np.nan
            ndvi[ndvi<=0.6]= 1
            
            nbr = (nir - swir2) / (nir + swir2)
            nbr[nbr >= 0.2]=np.nan
            nbr[nbr <= -0.05]=2
            nbr[nbr < 0.2]=1
            
            nbrfilter = nbr*swm*white*ndvi
                        
            ArraytoRaster(nbrfilter, NBRFILTER)
        
        print("___ for " + date + ": NBR raster created and filtered with SWM NDVI and RGB")
        
        ## Mask the results with CLC2018 PRIME2 WFD and home-made constant false alarm mask
        
        Local_mask = "E:/Masks_Landsat/" + tile + "_mask.shp"
        NBR_masked =  workplace + tile + "_" + date + "_NBR_masked.tif"
        ClipRwithS(Local_mask, NBRFILTER, NBR_masked)
        
        # Select only areas over 4000m2 
        
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
        seedintersection (NBRFILTER, resultsPoly, 1.0)
        finalraster = workplace + tile + "_" + date + "_final.tif"
        if not path.exists(finalraster):
            ras_ds = gdal.Open(NBRFILTER) 
            vec_ds = ogr.Open(resultsPoly) 
            lyr = vec_ds.GetLayer() 
            geotr = ras_ds.GetGeoTransform() 
            drv_tiff = gdal.GetDriverByName("GTiff") 
            chn_ras_ds = drv_tiff.Create(finalraster, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
            chn_ras_ds.SetGeoTransform(geotr)
            gdal.RasterizeLayer(chn_ras_ds, [1], lyr, options=['ATTRIBUTE=DN']) 
            chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0) 
            chn_ras_ds = None
        
        print("\n => " + date + ": burn area delineation done \n") 

print("----------------------------------------")
print("\n FINAL STEP")

imgList = os.listdir(workplace)
years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
months = ["01","02","03","04","05","06"]
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