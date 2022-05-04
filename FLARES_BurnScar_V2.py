# -*- coding: utf-8 -*-

# Author: Emma Chalençon
# Creation date: 27/08/2021
# Python version 3.8.5

# 06 October 2021: Modified by Guy Serbin for Mundi Linux environment:
    # Inclusion of IEO library variables and functions
    # Support for S3 object storage 

# 02 May 2022: 
    # Final modifications to version 1.0
    # Includes calculation of cloud-free rasters
    # Start of version 1.1, with improved memory management


# Version 1.1

import os, sys, fiona, rasterio, argparse, glob, datetime
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
# import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr
from shapely import wkt
from shapely.wkt import loads

try: # This is included as the module may not properly install in Anaconda.
    import ieo
except:
    # ieodir = os.getenv('IEO_INSTALLDIR')
    # if not ieodir:
    if os.path.isfile('../ieo/ieo.py'):
        ieodir = '../ieo'
    else:
        print('Error: IEO failed to load. Please input the location of the directory containing the IEO installation files.')
        ieodir = input('IEO installation path: ')
    if os.path.isfile(os.path.join(ieodir, 'ieo.py')):
        sys.path.append(ieodir)
        import ieo
        import S3ObjectStorage
    else:
        print('Error: that is not a valid path for the IEO module. Exiting.')
        sys.exit()


workplace = "/data/temp" #workplace directory
errorfile = os.path.join(ieo.logdir, 'FLARES_BurnScar_Errors.log')
srs = osr.SpatialReference()   
i = ieo.prjstr.find(':') + 1              
srs.ImportFromEPSG(int(ieo.prjstr[i:]))

def calcFilteredNBR(f, datadict, sensordict, *args, **kwargs):
    useqamask = kwargs.get('useqamask', False)
    qafile = kwargs.get('qafile', None)
    calccloudfree = kwargs.get('calccloudfree', True)
    basename = os.path.basename(f)[:-4]
    acqtime = ieo.envihdracqtime(f.replace('.dat', '.hdr'))
    prasters = ieo.envihdrparentrasters(f.replace('.dat', '.hdr'))
    
    if basename.endswith('_ref'):
        basename = basename[:-4]
    sceneid = basename
    ProductID = None
    parentrasters = []
    
    if prasters:
        i = prasters.index('{') + 1
        j = prasters.index('}')
        prasters = prasters[i : j].strip()
        parentrasters.append(prasters)
        if '.' in prasters:
            j = prasters.index('.')
            ProductID = prasters[:j]
        else:
            ProductID = sceneid
    else:
        ProductID = sceneid
    
    parts = sceneid.split('_')
    year, month, day, tile = parts[1][:4], parts[1][4:6], parts[1][6:], parts[2]
    
    if basename.startswith('S2'):
        key1 = 'Sentinel2'
        key2 = 'bands'
        ndvidir = ieo.Sen2ndvidir
        srdir = ieo.Sen2srdir
        outdir = os.path.join(workplace, 'sentinel2', tile)
    else:
        key1 = 'Landsat'
        # ndvidir = ieo.ndvidir
        srdir = ieo.srdir
        outdir = os.path.join(workplace, 'landsat', tile)
        if basename[2:3] in ['8', '9']:
            key2 = '8-9'
        else:
            key2 = '4-7'
    if not os.path.isdir(outdir):
        print(f'Creating output directory: {outdir}')
    # nbrdir = ndvidir.replace('NDVI', 'NBR')
    # ndvifile = os.path.join(ndvidir, f'{basename}.dat')
    # nbrfile = os.path.join(nbrdir, f'{basename}.dat')
    # if not os.path.isfile(nbrfile):
    #     print('NBR file is missing, calculating.')
    #     calcNDs(f, sensordict, calcNDTI = False)
        # if basename.startswith('S'):
        #     bucket = 'sentinel2'
        # else:
        #     bucket = 'landsat'
        # year, month, day, tile = basename[4:8], basename[8:10], basename[10:12], basename[13:16]
        # if not 'NBR' in datadict[tile][year][month][day][bucket].keys():
        #     datadict[tile][year][month][day][bucket]['NBR'] = {}
        #     datadict[tile][year][month][day][bucket]['NBR']['local'] = []
        # for f1 in [nbrfile, nbrfile.replace('.dat', '.hdr')]:
        
            
        #     basename1 = os.path.basename(f1)
        #     targetfilename = f'NBR/{tile}/{year}/{basename1}'
        #     targetdir = os.path.dirname(targetfilename)
        #     if not 'remote' in datadict[tile][year][month][day][bucket]['NBR'].keys():
        #         datadict[tile][year][month][day][bucket]['NBR']['remote'] = []
                
        #     if not targetfilename in datadict[tile][year][month][day][bucket]['NBR']['remote']:
        #         S3ObjectStorage.copyfilestobucket(bucket = bucket, filename = f1, targetdir = targetdir)
        #         datadict[tile][year][month][day][bucket]['NBR']['remote'].append(targetfilename) 
    if basename.startswith('S2'):
        key1 = 'Sentinel2'
        key2 = 'bands'
        # ndvidir = ieo.Sen2ndvidir
    else:
        key1 = 'Landsat'
        # ndvidir = ieo.ndvidir
        if basename[2:3] in ['8', '9']:
            key2 = '8-9'
        else:
            key2 = '4-7'
    # Open the reflectance file and check the number of bands
    refobj = gdal.Open(f)
    numbands = refobj.RasterCount
    
    # Check to see if the fullset of Sentinel-2 reflectance bands are present. 
    # If a subset based upon Landsat TM/ETM+ or OLI is present, use those band 
    # defaults.
    if key1 == 'Sentinel2' and numbands == 6: 
        key1 = 'Landsat'
        key2 = '4-7'
    elif key1 == 'Sentinel2' and numbands == 7:
        key1 = 'Landsat'
        key2 = '8-9'
    
    # Select bands
    blueband = sensordict[key1][key2]['blue']
    greenband = sensordict[key1][key2]['green']
    redband = sensordict[key1][key2]['red'] 
    nirband = sensordict[key1][key2]['nir']
    swir1band = sensordict[key1][key2]['swir1']
    swir2band = sensordict[key1][key2]['swir2'] 
    
    # for b in [f, nbrfile, ndvifile]:
    c, d = os.path.split(f)
    e = os.path.basename(c)
    parentrasters.append(f'{e}/{d}')
    # if useqamask:
    #     if not os.path.isfile(qafile):
    #         usefmask = False
    #         usecfmask = False
    # if usefmask or usecfmask:
    #     usefmask = True
    #     if not os.path.isfile(fmaskfile):
    #         fmaskfile = fmaskfile.replace('_cfmask.dat', '_fmask.dat')
    #         if not os.path.exists(fmaskfile):
    #             print('ERROR: Fmask file does not exist, returning.')
    #             logerror(fmaskfile, 'File not found.')
    #             usefmask = False
    #         else:
    #             parentrasters.append(os.path.basename(fmaskfile))

    # NDVIobj = gdal.Open(ndvifile)
    # ndvi = NDVIobj.GetRasterBand(1).ReadAsArray()
    # NBRobj = gdal.Open(nbrfile)
    # nbr = NBRobj.GetRasterBand(1).ReadAsArray()
    refobj = gdal.Open(f)

    # Get file geometry
    geoTrans = refobj.GetGeoTransform()
    ns = refobj.RasterXSize
    nl = refobj.RasterYSize
    if useqamask:
        if sceneid[2:3] == '0':
            landsat = int(sceneid[3:4])
        else:
            landsat = int(sceneid[2:3])
        fmask = ieo.maskfromqa_c2(qafile, landsat, sceneid)
    # elif usefmask:
    #     fmaskobj = gdal.Open(fmaskfile)
    #     fmaskdata = fmaskobj.GetRasterBand(1).ReadAsArray()
    #     fmask = numpy.zeros((nl, ns), dtype = numpy.uint8)
    #     maskvals = numexpr.evaluate('(fmaskdata == 0)')
    #     fmask[maskvals] = 1
    #     fmaskdata = None
    #     maskvals = None
    else:
        print('Warning: No Fmask file found for scene {}.'.format(sceneid))
        fmask = None
    
    blue = refobj.GetRasterBand(blueband).ReadAsArray()
    green = refobj.GetRasterBand(greenband).ReadAsArray()
    red = refobj.GetRasterBand(redband).ReadAsArray()
    nir = refobj.GetRasterBand(nirband).ReadAsArray()
    swir1 = refobj.GetRasterBand(swir1band).ReadAsArray()
    swir2 = refobj.GetRasterBand(swir2band).ReadAsArray()
    
    swir1 = swir1.astype('f4')
    blue = blue.astype('f4')
    green = green.astype('f4')
    red = red.astype('f4')
    nir = nir.astype('f4')
    swir2 = swir2.astype('f4')
    # ndvi = ndvi.astype('f4') / 10000
    # nbr = nbr.astype('f4') / 10000
    np.seterr(divide = 'ignore', invalid = 'ignore')
    
    white = blue + green + red
    white[white >= 2000] = np.nan
    white[white < 2000] = 1
    
    swm = (blue + green) / (nir + swir1)   #cloud and water mask
    swm[swm >= 0.4] = np.nan
    swm[swm < 0.4] = 1
    
    if calccloudfree:
        CLOUDFree = os.path.join(outdir, f'{tile}_{year}{month}{day}_CloudFree.tif')
        ArraytoRaster(swm, CLOUDFree, geoTrans)
        print(f"___ Cloud-free raster created and filtered with SWM NDVI and RGB for tile {tile} on date: {year}/{month}/{day}")
    
    ndvi = (nir - red) / (nir + red)
    ndvi_export = ndvi * swm
    print('Calculating filtered NDVI for scene {}.'.format(sceneid))
    NDVI = os.path.join(outdir, f'{basename}_NDVI.dat')
    ieo.ENVIfile(ndvi_export, 'NDVI', outdir = outdir, geoTrans = geoTrans, SceneID = sceneid, acqtime = acqtime, parentrasters = parentrasters, ProductID = ProductID, outfilename = NDVI).Save()
    # ArraytoRaster(ndvi_export, NDVI)
    
    ndvi[ndvi >= 0.6] = np.nan
    ndvi[ndvi <= 0.25] = np.nan
    ndvi[ndvi <= 0.6] = 1
  
    nbr = (nir - swir2) / (nir + swir2)
    nbr[nbr >= 0.01] = np.nan
    nbr[nbr <= -0.05] = 2
    nbr[nbr < 0.01] = 1
    
    print('Calculating filtered NBR for scene {}.'.format(sceneid))
    NBRFILTER = os.path.join(outdir, f'{basename}_nbrfilter.dat')
    nbrfilter = nbr * swm * white * ndvi
    ieo.ENVIfile(nbrfilter, 'NBR', outdir = outdir, geoTrans = geoTrans, SceneID = sceneid, acqtime = acqtime, parentrasters = parentrasters, ProductID = ProductID, outfilename = NBRFILTER).Save()

    
    refobj = None
    # NBRobj = None
    # NDVIobj = None
    nbrfilter = None
    blue = None
    green = None
    red = None
    nir = None
    swir1 = None
    ndvi = None
    nbr = None
    nbrfilter = None
    white = None 
    swm = None
    
    return NBRFILTER, NDVI, datadict
    

def calcNDs(f, sensordict, *args, **kwargs):
    calcNDTI = kwargs.get('calcNDTI', True)
    calcNBR = kwargs.get('calcNBR', True)
    baseoutputdir = kwargs.get('baseoutputdir', None)
    qafile = kwargs.get('qafile', None)
    useqamask = kwargs.get('useqamask', False)
    sceneid = kwargs.get('sceneid', None)
    ProductID = kwargs.get('ProductID', None)
    satellite = kwargs.get('satellite', None)
    useNTS = kwargs.get('useNTS', True)
    outfilebasename = None
    
    
    basename = os.path.basename(f)[:-4]
    acqtime = ieo.envihdracqtime(f.replace('.dat', '.hdr'))
    prasters = ieo.envihdrparentrasters(f.replace('.dat', '.hdr'))
    parentrasters = []
    if prasters:
        i = prasters.index('{') + 1
        j = prasters.index('}')
        prasters = prasters[i:j].strip()
        parentrasters.append(prasters)
        if '.' in prasters:
            j = prasters.index('.')
            ProductID = prasters[:j]
        
    if basename.endswith('_ref'):
        basename = basename[:-4]
    if basename.startswith('S2'):
        key1 = 'Sentinel2'
        key2 = 'bands'
        ndvidir = ieo.Sen2ndvidir
    else:
        key1 = 'Landsat'
        ndvidir = ieo.ndvidir
        if basename[2:3] in ['8', '9']:
            key2 = '8-9'
        else:
            key2 = '4-7'
    nirband = sensordict[key1][key2]['nir']
    swir1band = sensordict[key1][key2]['swir1']
    swir2band = sensordict[key1][key2]['swir2']
    
    if not baseoutputdir: #for NTS tiles
        NBRdir = ndvidir.replace('NDVI', 'NBR')
        NDTIdir = ndvidir.replace('NDVI', 'NDTI')
        NBRfile = os.path.join(NBRdir, f'{basename}.dat')
        NDTIfile = os.path.join(NDTIdir, f'{basename}.dat')
    else:
        NBRdir = os.path.join(baseoutputdir, 'NBR')
        NDTIdir = os.path.join(baseoutputdir, 'NDTI')
        NBRfile = os.path.join(NBRdir, f'{basename}_NBR.dat')
        NDTIfile = os.path.join(NDTIdir, f'{basename}_NDTI.dat')
    if calcNBR:
        if not os.path.isdir(NBRdir):
            print(f'Creating directory: {NBRdir}')
            os.makedirs(NBRdir)
    if calcNDTI:
        if not os.path.isdir(NDTIdir):
            print(f'Creating directory: {NDTIdir}')
            os.makedirs(NDTIdir)
    
    if not sceneid:
        # i = basename.find('.')
        if ProductID:
            sceneid = ProductID
        else:
            sceneid = basename # This will now use either the SceneID or ProductID
    
    # fmaskfile = os.path.join(fmaskdir,'{}_cfmask.dat'.format(sceneid))
    parentrasters.append(basename)
    # if useqamask:
    #     if not os.path.isfile(qafile):
    #         usefmask = False
    #         usecfmask = False
    # if usefmask or usecfmask:
    #     usefmask = True
    #     if not os.path.isfile(fmaskfile):
    #         fmaskfile = fmaskfile.replace('_cfmask.dat', '_fmask.dat')
    #         if not os.path.exists(fmaskfile):
    #             print('ERROR: Fmask file does not exist, returning.')
    #             logerror(fmaskfile, 'File not found.')
    #             usefmask = False
    #         else:
    #             parentrasters.append(os.path.basename(fmaskfile))

    
    refobj = gdal.Open(f)

    # Get file geometry
    geoTrans = refobj.GetGeoTransform()
    ns = refobj.RasterXSize
    nl = refobj.RasterYSize
    if useqamask:
        if sceneid[2:3] == '0':
            landsat = int(sceneid[3:4])
        else:
            landsat = int(sceneid[2:3])
        fmask = ieo.maskfromqa_c2(qafile, landsat, sceneid)
    # elif usefmask:
    #     fmaskobj = gdal.Open(fmaskfile)
    #     fmaskdata = fmaskobj.GetRasterBand(1).ReadAsArray()
    #     fmask = numpy.zeros((nl, ns), dtype = numpy.uint8)
    #     maskvals = numexpr.evaluate('(fmaskdata == 0)')
    #     fmask[maskvals] = 1
    #     fmaskdata = None
    #     maskvals = None
    else:
        print('Warning: No Fmask file found for scene {}.'.format(sceneid))
        fmask = None
    
    NIR = refobj.GetRasterBand(nirband).ReadAsArray()
    SWIR1 = refobj.GetRasterBand(swir1band).ReadAsArray()
    SWIR2 = refobj.GetRasterBand(swir2band).ReadAsArray()
    

    # NBR calculation
    if calcNBR:
        print('Calculating NBR for scene {}.'.format(sceneid))
        NBR = ieo.NDindex(NIR, SWIR2, fmask = fmask)
        parentrasters = ieo.makeparentrastersstring(parentrasters)
        ieo.ENVIfile(NBR, 'NBR', outdir = NBRdir, geoTrans = geoTrans, SceneID = sceneid, acqtime = acqtime, parentrasters = parentrasters, ProductID = ProductID, outfilename = NBRfile).Save()
        NBR = None
    if calcNDTI:
        print('Calculating NDTI for scene {}.'.format(sceneid))
        NDTI = ieo.NDindex(SWIR1, SWIR2, fmask = fmask)
        parentrasters = ieo.makeparentrastersstring(parentrasters)
        ieo.ENVIfile(NDTI, 'NDTI', outdir = NDTIdir, geoTrans = geoTrans, SceneID = sceneid, acqtime = acqtime, parentrasters = parentrasters, ProductID = ProductID, outfilename = NDTIfile).Save()
        NDTI = None
    NIR = None
    SWIR1 = None
    SWIR2 = None
    refobj = None
    fmask = None
    fmaskobj = None

def reprojection(raster, tile, date, bandN, dst_crs):
    """
    Function that reprojects a raster in the crs given by the variable dst_crs

    """
    proj_imagery = os.path.join(workplace, f'{tile}_{date}_{bandN}.tif')
    if not os.path.exists(proj_imagery):
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
                    source = rasterio.band(raster, i),
                    destination = rasterio.band(dst, i),
                    raster_transform = raster.transform,
                    raster_crs = raster.crs,
                    dst_transform = transform,
                    dst_crs = dst_crs,
                    resampling = Resampling.nearest)
    new_image = rasterio.open(proj_imagery)
    return new_image

def ArraytoRaster(array, outputRaster, geotransform):
    """
    Function that writes an array to a raster file

    """
    if not os.path.exists(outputRaster):
        ncols, nrows = np.shape(array)
        driver = gdal.GetDriverByName('GTiff')
        outputRaster= driver.Create(outputRaster, nrows, ncols, 1, gdal.GDT_Float64)
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
    if not os.path.exists(shapefile):
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
    if not os.path.exists(outraster):
        with fiona.open(shapefile, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes) #, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform})
        with rasterio.open(outraster, "w", **out_meta) as dest:
            dest.write(out_image)

# def ClipSwithS (Clipped, Clipper, New):
#     """
#     Function that clips a shapefile with another shapefile

#     """
#     if not os.path.exists(New):
#         Clipper = gpd.read_file(Clipper)
#         Clipped = gpd.read_file(Clipped)
#         clipped = gpd.clip(Clipped,Clipper)
#         clipped.to_file(New)
#     return New
    
def FilterPolyArea (shapefile, areaN):
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
        if area <= areaN:
            layer.DeleteFeature(feature.GetFID())
            continue
        feature.SetField("Area", area)
        layer.SetFeature(feature)
    
    j = 0    
    for feature in layer :
        j += 1
     
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

def setFeatureStats(fid, max, names = ["max", "fid"]):
    """
    Function that calculates stats per feature 

    """
    featstats = {names[0] : max, names[1] : fid,}
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
            offsets[3] - 1 - offsets[2], \
            offsets[1] - 1 - offsets[0], \
            1, \
            gdal.GDT_Byte)
             
            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values = [1])
            tr_array = tr_ds.ReadAsArray()
             
            r_array = r_ds.GetRasterBand(1).ReadAsArray(\
            offsets[2],\
            offsets[0],\
            offsets[3] - 1 - offsets[2],\
            offsets[1] - 1 - offsets[0])
             
            id = p_feat.GetFID()
            
            r_array[np.isnan(r_array)] = 0
            
            if r_array is not None:
                maskarray = np.ma.MaskedArray(\
                r_array,\
                mask = np.logical_or(r_array == nodata, np.logical_not(tr_array)))
                 
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
        i += 1
        if feature.GetFID() in deleteList:
            lyr.DeleteFeature(feature.GetFID())
    j = 0    
    for feature in lyr :
        j += 1
     
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
            offsets[3] - 1 - offsets[2], \
            offsets[1] - 1 - offsets[0], \
            1, \
            gdal.GDT_Byte)
             
            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values = [1])
            tr_array = tr_ds.ReadAsArray()
             
            r_array = r_ds.GetRasterBand(1).ReadAsArray(\
            offsets[2],\
            offsets[0],\
            offsets[3] - 1 - offsets[2],\
            offsets[1] - 1 - offsets[0])
             
            id = p_feat.GetFID()
            r_array[pd.isnull(r_array)] = 0
            
            if r_array is not None:
                maskarray = np.ma.MaskedArray(\
                r_array,\
                mask=np.logical_or(r_array == nodata, np.logical_not(tr_array)))
                 
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
                datestamp = date_raster[-18 : -10]
                feature.SetField("Date", datestamp)
                lyr.SetFeature(feature)
            
    r_ds = None
    p_ds = None

def ClipRwithR(extent_raster, original_raster, clipped_raster):
    """
    Function that clips a raster with another raster's extent

    """
    if not os.path.exists(clipped_raster):
        data = gdal.Open(extent_raster)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        gdal.Translate(clipped_raster, original_raster, projWin = [minx, maxy, maxx, miny])

def DeleteifCentroidin(file, mask, *args, **kwargs):
    sensor = kwargs.get('sensor', None)
    tile = kwargs.get('tile', None)
    if not os.path.isfile(file):
        file = os.path.join(workplace, sensor, tile, file)
    driver_file = ogr.GetDriverByName("ESRI Shapefile")
    data_file = driver_file.Open(file, 1)
    layer_file = data_file.GetLayer()        
    driver_mask = ogr.GetDriverByName('ESRI Shapefile')
    data_mask = driver_mask.Open(mask, 0)
    layer_mask = data_mask.GetLayer()
    i = 0
    for feature in layer_file :
        geom = feature.GetGeometryRef()
        Centroid = loads(geom.ExportToWkt()).representative_point()
        # print (Centroid)
        for polygons in layer_mask :
            geom1 = polygons.GetGeometryRef()
            geom1 = geom1.ExportToWkt()
            geom1 = wkt.loads(geom1)
            if Centroid.within(geom1):
                # print ('inside')
                # print (feature.GetFID())
                layer_file.DeleteFeature(feature.GetFID())
                i += 1
    data_file = None
    layer_file = None
    data_mask = None
    layer_mask = None
                
def CreateConstantFalseAlarmsMask(tile, sensor, *args, **kwargs):
    """
    Function that create a constant false alarms mask
    31 January 2022: function updated so final array values can be modified without hard coding

    """
    nancutoff = kwargs.get('nancutoff', 4)
    val1 = kwargs.get('val1', 1)
    val2 = kwargs.get('val2', 2)
    
    # Parameters
    tile_workplace = os.path.join(workplace, sensor, tile)
    # srs = osr.SpatialReference()                 
    # srs.ImportFromEPSG(2157)
    
    imgList = os.listdir(tile_workplace)
    img_list = []
    array_list = []
    
    for img in imgList:    
        if img.endswith("_ADDpoly.tif"):
            img_list.append(img)
            imgfile = os.path.join(tile_workplace, img)
            img_ds = gdal.Open(imgfile).ReadAsArray()
            dataset = gdal.Open(imgfile)
            geotransform = dataset.GetGeoTransform()
            img_ds[img_ds == np.nan] = 0
            array_list.append(img_ds)
    
    if len(array_list) >= 1 :
        array_out = np.nansum(array_list, axis = 0)
        finalArray = array_out 
        finalArray[finalArray <= nancutoff] = np.nan
        finalArray[finalArray <= val1] = 1
        finalArray[finalArray >= val2] = 2
        addRaster = os.path.join(tile_workplace, f"{sensor}_{tile}_FAADD.tif")
        if not os.path.exists(addRaster):
            ncols, nrows = np.shape(finalArray)
            driver = gdal.GetDriverByName('GTiff')
            outputRaster = driver.Create(addRaster, nrows, ncols, 1, gdal.GDT_Float64)
            outputRaster.SetGeoTransform(geotransform)
            outband = outputRaster.GetRasterBand(1)
            outband.WriteArray(finalArray)                    
            outputRaster.SetProjection(srs.ExportToWkt())
            outputRaster.FlushCache()
            outputRaster = None
        # print("___", len(array_list), "results combined")
        
        finalArray[finalArray <= 2] = 2
        addRasterpoly = os.path.join(tile_workplace, f"{sensor}_{tile}_FAADDpoly.tif")
        if not os.path.exists(addRasterpoly):
            ncols, nrows = np.shape(finalArray)
            driver = gdal.GetDriverByName('GTiff')
            outputRaster= driver.Create(addRasterpoly, nrows, ncols, 1, gdal.GDT_Float64)
            outputRaster.SetGeoTransform(geotransform)
            outband = outputRaster.GetRasterBand(1)
            outband.WriteArray(finalArray)                    
            outputRaster.SetProjection(srs.ExportToWkt())
            outputRaster.FlushCache()
            outputRaster = None
        finalPoly = os.path.join(tile_workplace, f"{sensor}_{tile}__FAmask.shp")
        if not os.path.exists(finalPoly):
            Polygonize (addRasterpoly, finalPoly)
        # print("___ and polygonized")
        
        print("\n" + "___ Polygons containing pixels with at least 4 repetitions considered constant false alarms")
        seedintersection (addRaster, finalPoly, 1.0) 
        return finalPoly

def dlDailyDataFromBucket(tile, sensor, *args, **kwargs):
    prefix = kwargs.get('prefix', 'Results/Sentinel2/')
    bucket = kwargs.get('bucket', 'wp3.1')
    outdir = kwargs.get('outdir', os.path.join(workplace, sensor, tile))
    prefix = f'{prefix}{tile}/'
    wp31tiledict = {}
    years = S3ObjectStorage.getbucketfoldercontents(bucket, prefix, '/')
    if len(years) > 2:
        for year in years:
            if not year in ['year_summary', 'FAmasks']:
                wp31tiledict[year] = {}
                months = S3ObjectStorage.getbucketfoldercontents(bucket, f'{prefix}{year}/', '/')
                for month in months:
                    wp31tiledict[year][month] = {}
                    days = S3ObjectStorage.getbucketfoldercontents(bucket, f'{prefix}{year}/{month}/', '/')
                    for day in days:
                        wp31tiledict[year][month][day] = S3ObjectStorage.getbucketfoldercontents(bucket, f'{prefix}{year}/{month}/{day}/', '')
                        if len(wp31tiledict[year][month][day]) > 0:
                            for f in wp31tiledict[year][month][day]:
                                outf = os.path.join(outdir, os.path.basename(f))
                                if (not os.path.isfile(outf)) and (f.endswith('_CloudFree.tif') or f.endswith('_final.tif') or f.endswith('_NDVI.dat') or f.endswith('_NDVI.hdr')):
                                    S3ObjectStorage.downloadfile(outdir, bucket, f)
    return wp31tiledict
                        
def createBuffer0(inputfn, outputBufferfn):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()
    # srs = osr.SpatialReference()                 
    # srs.ImportFromEPSG(2157)
    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, srs, geom_type = ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()
    new_field1 = ogr.FieldDefn('Date', ogr.OFTString)
    bufferlyr.CreateField(new_field1)
    
    for feature in inputlyr:
        date = feature.GetField("Date")
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(0)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        outFeature.SetField("Date", date)
        bufferlyr.CreateFeature(outFeature)


def main(startyear = 2015, endyear = 2021, startmonth = 1, endmonth = 6, \
        tile = None, sensor = 'both', useMGRS = False, verbose = False, \
        remove = False, transfer = False, nancutoff = 4, val1 = 7, val2 = 8, \
        excludeyearlist = '2015,2016,2017', ignoreprocessed = False, calccloudfree = True):
    
    excludeyearlist = excludeyearlist.split(',')
    # if len(excludeyearlist) > 0:
    #     for i in range(len(excludeyearlist)):
    #         excludeyearlist[i] = int(excludeyearlist[i])
    # Parameters
    # S2tiles = ieo.Sen2tiles # ["29UMT","28UGC","28UGD","29ULT","29UMA","29UMS","29UMT","29UMU","29UMV",
                 # "29UNA","29UNB","29UNT","29UNU","29UNV","29UPA","29UPB","29UPT","29UPU",
                 # "29UPV","29UQU","30UUE"]
    sensordict = {
                'Sentinel2' : {
                    'bands' : {
                        'blue' : 2,
                        'green' : 3,
                        'red' : 4,
                        'nir' : 8,
                        'swir1' : 11,
                        'swir2' : 12
                        }
                    },
                'Landsat' : {
                    '4-7' : {
                        'blue' : 1,
                        'green' : 2,
                        'red' : 3,
                        'nir' : 4,
                        'swir1' : 5,
                        'swir2' : 6
                        },
                    '8-9' : {
                        'blue' : 2,
                        'green' : 3,
                        'red' : 4,
                        'nir' : 5,
                        'swir1' : 6,
                        'swir2' : 7
                        },
                    },
                }
    transferdict = {}
    
    tfile = os.path.join(workplace, sensor.lower(), 'transferred_tiles.csv')
    processedtiles = []
    if os.path.isfile(tfile):
        with open(tfile, 'r') as lines:
            for line in lines:
                line = line.strip().split(',')
                for x in line:
                    if not x in processedtiles:
                        processedtiles.append(x)
    
    maskdir = os.path.join(workplace, 'masks')
    for d in [workplace, maskdir]:
        if not os.path.isdir(d):
            print(f'Creating directory: {d}')
            os.makedirs(d)
    # imgList = os.listdir(workplace) 
    
    # imgList = os.listdir(workplace)
    years = ['{}'.format(i) for i in range(startyear, endyear + 1)]
    months = ['{:02d}'.format(i) for i in range(1, 13)]
    minmonths = ['{:02d}'.format(i) for i in range(startmonth, endmonth + 1)] # These are used for calculating minimum NDVI
    if tile:
        if ',' in tile:
            tiles = tile.split(',')
        else:
            tiles = [tile]
    # elif useMGRS and sensor == 'sentinel2' and not tile:
    #     tiles = ieo.gettilelist(tiletype = 'sentinel2')
    else:
        tiles = ieo.gettilelist()
    # tilelist = [] # this is for the list of tiles that actually contain data to be processed, tiles is a list of all tiles.
    if ignoreprocessed:
        if len(processedtiles) > 0:
            print(f'{len(processedtiles)} have already be processed. Removing any processed tiles from processing list.')
            while any(tile in tiles for tile in processedtiles):
                for tile in tiles:
                    if tile in processedtiles:
                        print(f'Tile {tile} has already been processed, skipping.')
                        tiles.pop(tiles.index(tile))
    
    srs = osr.SpatialReference()   
    i = ieo.prjstr.find(':') + 1              
    srs.ImportFromEPSG(int(ieo.prjstr[i:]))
    
    dst_crs = ieo.prjstr # chosen coordinate reference system 
    
    # Read imagery files 
    if sensor == 'both':
        sensors = ['Sentinel2', 'Landsat']
        srdirs = [ieo.Sen2srdir, ieo.srdir]
        
    elif sensor.lower() == 'landsat':
        sensors = ['Landsat']
        srdirs = [ieo.srdir]
        slist = ['LE7', 'LC8', 'LC9']
    else:
        sensors = ['Sentinel2']
        slist = ['S2A', 'S2B']
        srdirs = [ieo.Sen2srdir]
    infilePath = "/data/temp/sentinel2" #imagery directory
    for tile in tiles:
        maskfilebasename = f'{tile}_mask.shp'
        if not os.path.isfile(os.path.join(maskdir, maskfilebasename)):
            print(f'Copying mask shapefile for tile {tile} to disk.')
            filelist = S3ObjectStorage.getFileList('airt.tiles.masks', prefix = tile)
            if len(filelist) > 0:
                for f in filelist:
                    S3ObjectStorage.downloadfile(maskdir, 'airt.tiles.masks', f)
            else:
                print(f'ERROR: There is no mask file for tile {tile}. Removing tile from processing list.')
                while tile in tiles:
                    tiles.pop(tiles.index(tile))
    # fileList = os.listdir(infilePath)
    # featuredict = ieo.getfeaturedict(tiletype = ieo.NTS) 
    
    if len(tiles) == 0:
        print('There are no tiles to process. Exiting.')
        sys.exit()
    
    datadict = {}
    for s in sensors:
        bucket = s.lower()
        srdir = srdirs[sensors.index(s)]
        print(f'Now processing data for {s}.')
        # Now create a dict of data stored locally and on S3 buckets.
        for tile in tiles:
            prefix = f'SR/{tile}/'
            if verbose: print(f'Scanning bucket {bucket}: SR/{tile}/')
            byears = S3ObjectStorage.getbucketfoldercontents(bucket, prefix, '/')
            if len(byears) > 0:
                for year in byears:
                    if verbose: print(f'Scanning {bucket}/SR/{tile}/{year}')
                    prefix = f'SR/{tile}/{year}/'
                    bmonths = S3ObjectStorage.getbucketfoldercontents(bucket, prefix, '/')
                    if len(bmonths) > 0:
                        for month in bmonths:
                            if verbose: print(f'Scanning {bucket}/SR/{tile}/{year}/{month}')
                            prefix = f'SR/{tile}/{year}/{month}/'
                            days = S3ObjectStorage.getbucketfoldercontents(bucket, prefix, '/')
                            if len(days) > 0:
                                for day in days:
                                    if verbose: print(f'Scanning {bucket}/SR/{tile}/{year}/{month}/{day}')
                                    for d in ['SR']: #, 'NDVI', 'NBR']:
                                        prefix = f'{d}/{tile}/{year}/{month}/{day}/'
                                        filelist = S3ObjectStorage.getbucketfoldercontents(bucket, prefix, '')
                                        if len(filelist) > 0:
                                            for f in filelist: 
                                                if f.endswith('.hdr') or f.endswith('.dat'):
                                                    if verbose: print(f'Adding remote file: {f}')
                                                    if not tile in datadict.keys():
                                                        datadict[tile] = {}
                                                    if not year in datadict[tile].keys():
                                                        datadict[tile][year] = {}
                                                    if not month in datadict[tile][year].keys():
                                                        datadict[tile][year][month] = {}
                                                    if not day in datadict[tile][year][month].keys():
                                                        datadict[tile][year][month][day] = {}
                                                    if not bucket in datadict[tile][year][month][day].keys():
                                                        datadict[tile][year][month][day][bucket] = {}
                                                    if not d in datadict[tile][year][month][day][bucket].keys():
                                                        datadict[tile][year][month][day][bucket][d] = {'remote' : []}
                                                    if not 'remote' in datadict[tile][year][month][day][bucket][d].keys():
                                                        datadict[tile][year][month][day][bucket][d] = {'remote' : []}
                                                    datadict[tile][year][month][day][bucket][d]['remote'].append(f)        
                                        
            # Now scan local directories for files
            if verbose: print('Scanning local directories for input files.')                                
            for d in ['SR']: #, 'NDVI', 'NBR']:
                if d != 'SR':
                    scandir = os.path.join(os.path.dirname(srdir), d)
                else:
                    scandir = srdir
                if verbose: print(f'Scanning for {d} files.')       
                if os.path.isdir(scandir):
                    filelist = glob.glob(os.path.join(scandir, f'*_{tile}.*'))
                    if len(filelist) > 0:
                        for f in filelist:
                            if not f.endswith('.bak'):
                                
                                if verbose: print(f'Found local {d} file: {f}')
                                basename = os.path.basename(f)[:16]
                                year, month, day, tile = basename[4:8], basename[8:10], basename[10:12], basename[13:16]
                                if year in years:
                                    # if not tile in tilelist:
                                    #     tilelist.append(tile)
                                    if not tile in datadict.keys():
                                        datadict[tile] = {}
                                    if not year in datadict[tile].keys():
                                        datadict[tile][year] = {}
                                    if not month in datadict[tile][year].keys():
                                        datadict[tile][year][month] = {}
                                    if not day in datadict[tile][year][month].keys():
                                        datadict[tile][year][month][day] = {}
                                    if not bucket in datadict[tile][year][month][day].keys():
                                        datadict[tile][year][month][day][bucket] = {}
                                    if not d in datadict[tile][year][month][day][bucket].keys():
                                        datadict[tile][year][month][day][bucket][d] = {}
                                    if not 'local' in datadict[tile][year][month][day][bucket][d].keys():
                                        datadict[tile][year][month][day][bucket][d]['local'] = []
                                    datadict[tile][year][month][day][bucket][d]['local'].append(f) 
                else:
                    print(f'Creating directory: {scandir}')
                    os.makedirs(scandir)
        # Begin processing the data
        if verbose: print('Total tiles to process: {}'.format(len(tiles)))
        if verbose: print('Now checking to make sure that input files are saved both remotely and locally, and if not, transfer to where needed after processing.')
        
        for tile in tiles:
            tile_start_time = datetime.datetime.now()
            if verbose: print(f'{tile_start_time.strftime("%Y/%m/%d %H:%M:%S")}: Processing tile {tile}.')    
            tile_workplace = os.path.join(workplace, sensor.lower(), tile)
            if not os.path.isdir(tile_workplace):
                os.makedirs(tile_workplace)
            if verbose: print(f'Downloading any data on bucket wp3.1 for tile {tile}.')
            wp31dldict = dlDailyDataFromBucket(tile, sensor)
            for year in years:
                if verbose: print(f'Processing year {year}.')
                for month in months:
                    if month in datadict[tile][year].keys():
                        if verbose: print(f'Processing month {month}.')
                        days = sorted(datadict[tile][year][month].keys())
                        for day in days:
                            processed = False
                            if verbose: print(f'Processing day {day}.')
                            for d in ['SR']: #, 'NDVI', 'NBR']:
                                if verbose: print(f'Processing data type: {d}.')
                                if d in datadict[tile][year][month][day][bucket].keys():
                                    # Check to see if local files are present on buckets. If not, copy them over.
                                    if 'local' in datadict[tile][year][month][day][bucket][d].keys(): 
                                        for f in datadict[tile][year][month][day][bucket][d]['local']:
                                            basename = os.path.basename(f)
                                            targetfilename = f'{d}/{tile}/{year}/{basename}'
                                            targetdir = os.path.dirname(targetfilename)
                                            if not 'remote' in datadict[tile][year][month][day][bucket][d].keys():
                                                datadict[tile][year][month][day][bucket][d]['remote'] = []
                                                
                                            if not targetfilename in datadict[tile][year][month][day][bucket][d]['remote']:
                                                if transfer:
                                                    print(f'Found file missing on bucket {bucket}, adding to transfer lists: {f}')
                                                    if not bucket in transferdict.keys():
                                                        transferdict[bucket] = {}
                                                    if not d in transferdict[bucket].keys():
                                                        transferdict[bucket][d] = {}
                                                    if not tile in transferdict[bucket][d].keys():
                                                        transferdict[bucket][d][tile] = {}
                                                    if not year in transferdict[bucket][d][tile].keys():
                                                        transferdict[bucket][d][tile][year] = []
                                                    transferdict[bucket][d][tile][year].append(f)
                                                    # S3ObjectStorage.copyfilestobucket(bucket = bucket, filename = f, targetdir = targetdir)
                                                    # datadict[tile][year][month][day][bucket][d]['remote'].append(targetfilename)
                                    else:
                                        datadict[tile][year][month][day][bucket][d]['local'] = []
                                    # Check to see if remote bucket files are present on locally. If not, copy them over.
                                    
                                    if year in wp31dldict.keys():
                                        if month in wp31dldict[year].keys():
                                            if day in wp31dldict[year][month].keys():
                                                if len(wp31dldict[year][month][day]) > 0: 
                                                    if any(x.endswith(f'{tile}_{year}{month}{day}_final.tif') for x in wp31dldict[year][month][day]) and any(x.endswith(f'{tile}_{year}{month}{day}_CloudFree.tif') for x in wp31dldict[year][month][day]):
                                                        processed = True
                                    if 'remote' in datadict[tile][year][month][day][bucket][d].keys(): 
                                        for f in datadict[tile][year][month][day][bucket][d]['remote']:
                                            if d != 'SR':
                                                localdir = os.path.join(os.path.dirname(srdir), d)
                                            else:
                                                localdir = srdir
                                            if not os.path.isdir(localdir):
                                                print(f'Creating local directory: {localdir}')
                                                os.makedirs(localdir)
                                            basename = os.path.basename(f)
                                            localf = os.path.join(localdir, basename)
                                            if (not localf in datadict[tile][year][month][day][bucket][d]['local']) and (not processed):
                                                print(f'Copying from bucket {bucket}: {f}')
                                                S3ObjectStorage.downloadfile(localdir, bucket, f)
                                                datadict[tile][year][month][day][bucket][d]['local'].append(localf)
                            
                            if not processed:
                                finalraster = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}{month}{day}_final.tif')
                                if any(x.endswith('.dat') for x in datadict[tile][year][month][day][bucket]['SR']['local']) and any(x.endswith('.hdr') for x in datadict[tile][year][month][day][bucket]['SR']['local']):
                                    # print(datadict[tile][year][month][day][bucket]['SR']['local'])
                                    SRfile = [x for x in datadict[tile][year][month][day][bucket]['SR']['local'] if x.endswith('.dat')][0]
                                else:
                                    for d in ['SR']: #, 'NDVI', 'NBR']:
                                        dlflist = S3ObjectStorage.getFileList(bucket, prefix = f'{d}/{tile}/{year}/{month}/{day}/')
                                        if len(dlflist) > 0:
                                            for dlfname in dlflist:
                                                S3ObjectStorage.downloadfile(srdir.replace('SR', d), bucket, dlfname)
                                                datadict[tile][year][month][day][bucket][d]['local'].append(os.path.join(os.path.dirname(srdir), d, os.path.basename(dlfname)))
                                            SRfile = [x for x in datadict[tile][year][month][day][bucket]['SR']['local'] if x.endswith('.dat')][0]
                                if not os.path.isfile(finalraster):   
                                    print("\n Let's start the delineation process:") 
                                    NBRFILTER, NDVI, datadict = calcFilteredNBR(SRfile, datadict, sensordict, calccloudfree = calccloudfree) 
                                    print(f"___ for {year}/{month}/{day}: NBR raster created and filtered with SWM NDVI and RGB")
                                    
                                    ## Mask the results with CLC2018 PRIME2 WFD and home-made constant false alarm mask
                                    
                                    Local_mask = os.path.join(maskdir, f'{tile}_mask.shp') #path the S2 masks 
                                    NBR_masked = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}{month}{day}_NBR_masked.tif')
                                    ClipRwithS(Local_mask, NBRFILTER, NBR_masked)
                                    
                                    ## Select only areas over 0.4ha 
                                    
                                    resultsRaster = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}{month}{day}_results.tif')
                                    if not os.path.isfile(resultsRaster): 
                                        res = gdal.Open(NBR_masked).ReadAsArray()
                                        dataset = gdal.Open(NBR_masked)
                                        geotransform = dataset.GetGeoTransform()
                                        res = res.astype('f4')
                                        res[res == 1] = 2
                                        ArraytoRaster(res, resultsRaster, geotransform)
                                        dataset = None 
                                    resultsPoly = resultsRaster[:-4] + "_poly.shp" 
                                    if not os.path.isfile(resultsPoly):
                                        resultsPoly = Polygonize (resultsRaster, resultsPoly) # This was "outPoly" in the original code, but appeared nowehere else
                                    print(f"___ for {year}/{month}/{day}: Filtering areas under 0.4 ha.")    
                                    FilterPolyArea (resultsPoly, 4000)
                                    
                                    ## Only keep polygons which contain a NBR_masked high intensity seed (<= -0.05)
                                    
                                    print(f"___ for {year}/{month}/{day}: Discarding polygons without high intensity seeds.")
                                    seedintersection (NBR_masked, resultsPoly, 1.0)
                                    
                                    ras_ds = gdal.Open(NBR_masked) 
                                    vec_ds = ogr.Open(resultsPoly) 
                                    lyr = vec_ds.GetLayer() 
                                    geot = ras_ds.GetGeoTransform() 
                                    drv_tiff = gdal.GetDriverByName("GTiff") 
                                    chn_ras_ds = drv_tiff.Create(finalraster, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
                                    chn_ras_ds.SetGeoTransform(geot)
                                    chn_ras_ds.SetProjection(srs.ExportToWkt())
                                    gdal.RasterizeLayer(chn_ras_ds, [1], lyr, options = ['ATTRIBUTE=DN']) 
                                    chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0) 
                                    chn_ras_ds.FlushCache()
                                    chn_ras_ds = None
                                
                                    print(f"\n => {year}/{month}/{day}: burn area delineation done \n") 
                                    if not 'filtered' in datadict[tile][year][month][day][bucket].keys():
                                        datadict[tile][year][month][day][bucket]['filtered'] = {}
                                        for key in ['final', 'NBR', 'NDVI']:
                                            datadict[tile][year][month][day][bucket]['filtered'][key] = []
                                    for d, f in zip(['final', 'NDVI', 'NBR'], [finalraster, NDVI, NBRFILTER]):
                                        datadict[tile][year][month][day][bucket]['filtered'][d].append(f)
                                    if ieo.useS3 and remove:
                                        for f in [SRfile, SRfile.replace('.dat', '.hdr')]:
                                            print(f'Deleting input file from disk: {f}')
                                            os.remove(f)
                
                print("----------------------------------------")
                print(f"\n {tile}/ {year}: FINAL STEP")
    # function
        # sensor/ bucket
            # tile
                # year
                array_out = []                
                filecounter = 0
                img_list = []
                cf_array_list = []
                cf_array = []
                Min = []
                Max = []
                # try:
                for month in datadict[tile][year].keys():
                    for day in datadict[tile][year][month].keys():
                        filelist = glob.glob(os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}{month}{day}_final.tif'))
                        if len(filelist) > 0:
                            for f in filelist:
                        ## Add all the burnscar polygons from the same tile/year:
                        # for f in datadict[tile][year][month][day][bucket]['filtered']['final']:
                                img_list.append(f)
                                img_ds = gdal.Open(f).ReadAsArray()
                                dataset = gdal.Open(f)
                                geotransform = dataset.GetGeoTransform()
                                img_ds[img_ds == np.nan] = 0
                                if len(array_out) == 0:
                                    array_out = img_ds
                                else:
                                    array_out = np.nansum([array_out, img_ds], axis = 0)
                                filecounter += 1
                                # array_list.append(img_ds)
                        
                        ## Calculate the difference between the min ndvi Jan-Jun/ the max ndvi Jul-Dec:
                        # if burn scar: positive dif of at least 0.1 but not higher than 0.6
                        if not year in excludeyearlist:
                            filelist = glob.glob(os.path.join(workplace, sensor.lower(), tile, f'*_{year}{month}{day}_{tile}_NDVI.dat'))
                            if len(filelist) > 0:
                                for f in filelist:
                            # for f in datadict[tile][year][month][day][bucket]['filtered']['NDVI']:
                                    img_ds = gdal.Open(f).ReadAsArray()
                                    dataset = gdal.Open(f)
                                    geotransform = dataset.GetGeoTransform()
                                    img_ds[img_ds == np.nan] = 0
                                    if month in minmonths:
                                        if len(Min) == 0:
                                            Min = img_ds
                                        else:
                                            Min = np.nanmin([Min, img_ds], axis = 0)
                                    else:
                                        if len(Max) == 0:
                                            Max = img_ds
                                        else:
                                            Max = np.nanmax([Max, img_ds], axis = 0)
                        
                        ## If calccloudfree = True, aggregate cloud-free image for year
                        if calccloudfree:
                            CLOUDFree = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}{month}{day}_CloudFree.tif') 
                            
                            if os.path.isfile(CLOUDFree):
                                img_ds = gdal.Open(CLOUDFree).ReadAsArray()
                                dataset = gdal.Open(CLOUDFree)
                                geotransform = dataset.GetGeoTransform()
                                img_ds[img_ds == np.nan] = 0
                                if len(cf_array) == 0:
                                    cf_array = img_ds
                                else:
                                    cf_array_list = [cf_array, img_ds]
                                    cf_array = np.nansum(cf_array_list, axis = 0)
                                    
                                # cf_array_list.append(img_ds) 
                                img_ds = None
                                cf_array_list = None
                                dataset = None
                                
        
                            
                # monthly and daily stacking completed
                
                ## If calccloudfree = True, create cloud-free image for year
                if calccloudfree:
                    if len(cf_array) > 0: # >= 1 :  
                        # array_out = np.nansum(cf_array_list, axis = 0)
                        # finalArray = array_out 
                        cf_array[cf_array <= 0] = np.nan
                        addRaster = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_CF_ADD.tif')
                        if not os.path.isfile(addRaster):
                            ArraytoRaster(cf_array, addRaster, geotransform)
                        print(f"______ {filecounter} cloud-free layer results combined for tile {tile} for the year {year}.")
                        cf_array = None
                        
                        
                
                if len(Min) >= 1:
                    # addin = np.stack(min_list, axis = 0 )
                    # Min = np.nanmin(addin, axis = 0)      
                    if len(Max) >= 1:
                        # addax = np.stack(max_list, axis = 0 )
                        # Max = np.nanmax(addax, axis = 0)    
                    
                        Diff = Max - Min  
                        MAX = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_Max.tif')
                        MIN = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_Min.tif')
                        if not os.path.exists(MIN):
                            ArraytoRaster(Min, MIN, geotransform)
                            ArraytoRaster(Max, MAX, geotransform)
                        Diff[Diff >= 0.6] = np.nan
                        Diff[Diff <= 0.1] = np.nan
                        Diff[Diff >= 0.0] = 1
                        Diff[Diff == np.nan] = np.nan
                        DIFF = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_Diff.tif')
                        if not os.path.exists(DIFF):
                            ArraytoRaster(Diff, DIFF, geotransform)    
                print(f"\n Tile {tile}: {filecounter} images analysed for {year}.") #.format(tile, len(array_list), year))            
                
                if len(array_out) >= 1:  
                    # array_out = np.nansum(array_list, axis = 0)
                    if not year in excludeyearlist:
                        finalArray = array_out * Diff
                    else:
                        finalArray = array_out
                    finalArray[finalArray <= 0] = np.nan
                    addRaster = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_ADD.tif')
                    if not os.path.exists(addRaster):
                        ArraytoRaster(finalArray, addRaster, geotransform)
                    print(f"___ {filecounter} results combined")
                    
                    finalArray[finalArray >= 2] = 2        
                    addRasterpoly = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_ADDpoly.tif')
                    if not os.path.exists(addRasterpoly):
                        ArraytoRaster(finalArray, addRasterpoly, geotransform)
                    finalPoly = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}.shp')
                    if not os.path.exists(finalPoly):
                        Polygonize (addRasterpoly, finalPoly)
                    print("___ and polygonized")
                    
                    
                    if not year in excludeyearlist:
                        seedintersection (addRaster, finalPoly, 3.0)
                        FilterPolyArea (finalPoly, 4000)
                        print("___ Polygons present on one single date only discarded")
                
                    img_list.sort()
                    for img in img_list:
                        timestamp(os.path.join(workplace, sensor.lower(), tile, img), finalPoly)
                    print("___ Polygons time-stamped")   
                    
                    print("\n----------------------------------------")

                    print("\n Final filtering")
                    
                    print("\n----------------------------------------")
                # except Exception as e:
                #     print(f'ERROR: {tile}: {e}')
                #     ieo.logerror(tile, e, errorfile = errorfile)        
            FA_mask = CreateConstantFalseAlarmsMask(tile, sensor, nancutoff = nancutoff, val1 = val1, val2 = val2)
            for year in years:
                yearshp = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}.shp')
                if os.path.isfile(yearshp):
                # imgList = os.listdir(os.path.join(workplace, sensor.lower(), tile))
                # for img in imgList:
                #     if img.endswith(f'{tile}_{year}.shp') and os.path.isfile(img):
                        print(f'Tile {tile}: {year} - filtering results for constant false alarms.')
                        DeleteifCentroidin (yearshp, FA_mask)
                        outputBufferfn = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_final.shp')
                        if not os.path.exists(outputBufferfn):
                            createBuffer0(yearshp, outputBufferfn)
                else: 
                    print(f'ERROR: File missing: {yearshp}')
                    ieo.logerror(tile, f'File missing: {yearshp}', errorfile = errorfile)
                        
            for year in years:
                print(f'Processing year: {year}')
                yearshp = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{year}_final.shp')
                mask = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{int(year) - 1}_final.shp')
                if year != '2015' and os.path.isfile(yearshp) and os.path.isfile(mask):
                # imgList = os.listdir(os.path.join(workplace, sensor.lower(), tile))
                # for img in imgList:
                #     if img.endswith(f'{tile}_{year}_final.shp') and year != "2015":
                        # mask = os.path.join(workplace, sensor.lower(), tile, f'{tile}_{int(year) - 1}_final.shp')
                        print(f'Tile {tile}: filtering results for scars appearing in both {int(year) - 1} and {year}.')
                        DeleteifCentroidin (yearshp, mask)
            print("______ Results filtered to not report a burnscar two years in a row")
            
            print("----------------------------------------")
                     
                    # print("\n----------------------------------------")
                    # print(f'Processing complete for year {year} for tile {tile}.')
                
            print("\n----------------------------------------")
            print(f'Processing complete for tile {tile}.')
            if remove:
                print(f'Now cleaning up local input files for tile {tile}.')
                # if bucket == 'sentinel2':
                #     infiledir = ieo.Sen2srdir
                # else:
                #     infiledir = ieo.srdir
                # infilelist = glob.glob(os.path.join(infiledir, f'S*_{tile}.*'))
                # in len(infilelist) > 0:
                for year in datadict[tile].keys():
                    for month in datadict[tile][year].keys():
                        for day in datadict[tile][year][month].keys():
                            for d in ['SR']: #, 'NDVI', 'NBR']:
                                if 'local' in datadict[tile][year][month][day][bucket][d].keys():
                                    for f in datadict[tile][year][month][day][bucket][d]['local']:
                                        if os.path.isfile(f):
                                            if verbose: print(f'Deleting: {f}')
                                            os.remove(f)
                print(f'Now moving processed files for tile {tile} to bucket wp3.1.')
                tiledict = {}
                tiledict['FAmasks'] = glob.glob(os.path.join(workplace, sensor.lower(), tile, 'sentinel2*.*'))
                tiledict['Year_summary'] = []
                for n in [tile, 'S2']:
                    flist = glob.glob(os.path.join(workplace, sensor.lower(), tile, f'{n}*.*'))
                    if len(flist) > 0:
                        for f in flist: 
                            basename = os.path.basename(f)
                            
                            i = os.path.basename(f).find('_')
                            if basename[8:9] in ['_', '.'] and basename[:8] != 'sentinel2':
                                tiledict['Year_summary'].append(f)
                            else:
                                parts = os.path.basename(f).split('_')
                                year, month, day = parts[1][:4], parts[1][4:6], parts[1][6:]
                                if not year in tiledict.keys():
                                    tiledict[year] = {}
                                if not month in tiledict[year].keys():
                                    tiledict[year][month] = {}
                                if not day in tiledict[year][month].keys():
                                    tiledict[year][month][day] = []
                                tiledict[year][month][day].append(f)
                for key in sorted(tiledict.keys()):
                    if isinstance(tiledict[key], list):
                        if len(tiledict[key]) > 0:
                            S3ObjectStorage.copyfilestobucket(filelist = tiledict[key], bucket = 'wp3.1', targetdir = f'Results/Sentinel2/{tile}/{key}')
                            for f in tiledict[key]:
                                print(f'Deleting from disk: {f}')
                                os.remove(f)
                    else:
                        for month in sorted(tiledict[key].keys()):
                            for day in sorted(tiledict[key][month].keys()):
                                transferfiles = True
                                if year in wp31dldict.keys():
                                    if month in wp31dldict[year].keys():
                                        if day in wp31dldict[year][month].keys():
                                            if len(wp31dldict[year][month][day]) > 0:
                                                if any(x.endswith(f'{tile}_{year}{month}{day}_final.tif') for x in wp31dldict[year][month][day]) and any(x.endswith(f'{tile}_{year}{month}{day}_CloudFree.tif') for x in wp31dldict[year][month][day]):
                                                    transferfiles = False
                                if transferfiles:
                                    S3ObjectStorage.copyfilestobucket(filelist = tiledict[key][month][day], bucket = 'wp3.1', targetdir = f'Results/Sentinel2/{tile}/{key}/{month}/{day}')
                                for f in tiledict[key][month][day]:
                                   try:
                                       print(f'Deleting from disk: {f}')
                                       os.remove(f)
                                   except Exception as e:
                                       print(f'ERROR: {e}')
                flist = glob.glob(os.path.join(workplace, sensor.lower(), tile, '*.*'))
                if len(flist) == 0:
                    print(f'Removing directory: {os.path.join(workplace, tile)}')
                    os.rmdir(os.path.join(workplace, sensor.lower(), tile))
                if not os.path.isfile(tfile):
                    with open(tfile, 'w') as output:
                        output.write(tile)
                else:
                    with open(tfile, 'a') as output:
                        output.write(f',{tile}')
            if verbose:
                tile_end_time = datetime.datetime.now()
                tile_timedelta = (tile_end_time - tile_start_time).seconds
                hours = int(tile_timedelta // 3600)
                minutes = int((tile_timedelta // 60) % 60)
                seconds = tile_timedelta % 3600
                print(f'{tile_end_time.strftime("%Y/%m/%d %H:%M:%D")}: Tile {tile} finished processing. Execution time: {hours}:{minutes}:{seconds:02f}')
        print("\n----------------------------------------")
        print(f'Processing complete for {sensor}.')
    # if transfer:
    
    print("\n----------------------------------------")
    print('Processing complete.')
       
        
                        
            
        
if __name__ == '__main__':
    # Parse the expected command line arguments
    parser = argparse.ArgumentParser('This script processes Landsat and Sentinel 2 data to determine burn scar areas.')
    parser.add_argument('--startyear', type = int, default = 2015, help = 'Start year for analysis, default = 2015.')
    parser.add_argument('--endyear', type = int, default = 2021, help = "End year for analysis, default = 2021.")
    parser.add_argument('--startmonth', type = int, default = 1, help = 'Start month for analysis, default = 1.')
    parser.add_argument('--endmonth', type = int, default = 6, help = "End month for analysis, default = 12.")
    parser.add_argument('-t', '--tile', type = str, default = None, help = 'Tile to process. Assumed to be in ieo.NTS.')
    parser.add_argument('--sensor', required = False, default = 'sentinel2', type = str, choices = ['both','landsat', 'sentinel2'], help='Process Landsat or Sentinel 2. Default = "both".')
    parser.add_argument('--useMGRS', action = 'store_true', help = 'Use MGRS tiles rather than those in ieo.NTS. Only valid if "--sensor=sentinel2".')
    parser.add_argument('--verbose', action = 'store_true', help = 'Display more messages during execution.')
    parser.add_argument('--remove', action = 'store_true', help = 'Delete local input files after processing.')
    parser.add_argument('--transfer', action = 'store_true', help = 'Transfer local files that are missing on remote buckets.')
    parser.add_argument('--nancutoff', type = int, default = 4, help = "Cutoff value, default = 4.")
    parser.add_argument('--val1', type = int, default = 7, help = "val1 value, default = 7.")
    parser.add_argument('--val2', type = int, default = 8, help = "val2 value, default = 8.")
    parser.add_argument('--excludeyearlist', type = str, default = '2015,2016,2017', help = 'Comma-delimited list of years to be excluded from differential analyses. Default = "2015,2016,2017".')
    parser.add_argument('--ignoreprocessed', type = bool, default = True, help = 'Do not process tiles which have already been processed.')
    parser.add_argument('--calccloudfree', type = bool, default = True, help = 'Save cloud-free images, default = True.')
    
    args = parser.parse_args()
 
    # Pass the parsed arguments to mainline processing   
    main(args.startyear, args.endyear, args.startmonth, args.endmonth, \
        args.tile, args.sensor, args.useMGRS, args.verbose, args.remove, \
        args.transfer, args.nancutoff, args.val1, args.val2, \
        args.excludeyearlist, args.ignoreprocessed, args.calccloudfree)
