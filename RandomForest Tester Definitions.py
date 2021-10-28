# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:51:25 2021

@author: Zam
"""


# Import Python 3's print function and division
from __future__ import print_function, division

# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array, ogr, gdal, gdalconst 
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import pandas as pd
import subprocess
import sys
import os


import tensorflow as tf

from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



# First setup a 5-4-3 composite
def color_stretch(image, index, minmax=(0, 10000)):
    colors = image[:, :, index].astype(np.float64)

    max_val = minmax[1]
    min_val = minmax[0]

    # Enforce maximum and minimum values
    colors[colors[:, :, :] > max_val] = max_val
    colors[colors[:, :, :] < min_val] = min_val

    for b in range(colors.shape[2]):
        colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)
        
    return colors





filepath = r'C:\Users\Zam\Desktop\Masters\EOanalytics\SR' #this is the filepath to where the Surface Reflectance datasets are located
output_path = r'C:\Users\Zam\Desktop\Masters\EOanalytics\Output' #this is the output folder
gorse_path = r'C:\Users\Zam\Desktop\Masters\EOanalytics\gorse' #this is the filepath to the "tester" shapefile, which is the shapefile which creates the training dataset
tester_shapefile =gorse_path + '\\'  + 'Gorse_Tester_2.shp'


entries = os.listdir(filepath)
#print(entries)
LC8 = []
LE7 = []
LC8K07 = []
LE7K07 = []
LC8J07 = []

for i in entries:
    if i.endswith(".dat") and i.startswith("LC8"):
        LC8.append(i)
    elif i.endswith(".dat") and i.startswith("LE7"):
        LE7.append(i)

for y in LC8:
    if y.find('K07') > 0:
        LC8K07.append(y)
    elif y.find('J07') > 0:
        LC8J07.append(y)

for y in LE7:
    if y.find('K07') > 0:
        LE7K07.append(y)


#print(LC8)
#print(LE7)
#print(LC8K07)



# Open the dataset from the file
dataset = ogr.Open(tester_shapefile)
# Make sure the dataset exists -- it would be None if we couldn't open it
if not dataset:
    print('Error: could not open dataset')
    
### Let's get the driver from this file
driver = dataset.GetDriver()
print('Dataset driver is: {n}\n'.format(n=driver.name))

### How many layers are contained in this Shapefile?
layer_count = dataset.GetLayerCount()
print('The shapefile has {n} layer(s)\n'.format(n=layer_count))

### What is the name of the 1 layer?
layer = dataset.GetLayerByIndex(0)
print('The layer is named: {n}\n'.format(n=layer.GetName()))

# layer = dataset.GetLayerByIndex(0)
# print('The layer is named: {n}\n'.format(n=layer.GetName()))

### What is the layer's geometry? is it a point? a polyline? a polygon?
# First read in the geometry - but this is the enumerated type's value
geometry = layer.GetGeomType()

# So we need to translate it to the name of the enum
geometry_name = ogr.GeometryTypeToName(geometry)
print("The layer's geometry is: {geom}\n".format(geom=geometry_name))

### What is the layer's projection?
# Get the spatial reference
spatial_ref = layer.GetSpatialRef()

# Export this spatial reference to something we can read... like the Proj4
proj4 = spatial_ref.ExportToProj4()
print('Layer projection is: {proj4}\n'.format(proj4=proj4))

### How many features are in the layer?
feature_count = layer.GetFeatureCount()
print('Layer has {n} features\n'.format(n=feature_count))

### How many fields are in the shapefile, and what are their names?
# First we need to capture the layer definition
defn = layer.GetLayerDefn()

# How many fields
field_count = defn.GetFieldCount()
print('Layer has {n} fields'.format(n=field_count))

# What are their names?
print('Their names are: ')
for i in range(field_count):
    field_defn = defn.GetFieldDefn(i)
    print('\t{name} - {datatype}'.format(name=field_defn.GetName(),
                                         datatype=field_defn.GetTypeName()))
    



########Turn dat file into gtif


gtif_file = gorse_path + '\\'+ 'prediction_temp.gtif'

#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\ieo_copy\LC8_2018272_K07.dat'
#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\SR\LC8_2016242_K07.dat'
#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\ieo_copy\LC8_2018183_K07.dat'
#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\ieo_copy\LC8_2018295_K07.dat'

#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\ieo_copy\LC8_2016242_J05.dat'
#vrt_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\SR\vrt\LC8_2017036.vrt'

#
#dat_file = r'C:\Users\Zam\Desktop\Masters\EOanalytics\ieo_copy\LC8_2018295_K07.dat'

def image_analyzer(dat_name, gtif_file, layer, bandcount = 8):
    
    dat_file = filepath + '\\'  + str(dat_name)
        
    crs =  'ESPG:4173'
    args = ['gdal_translate', '-of', 'Gtiff', dat_file , gtif_file]
    #args = ['gdal_translate', '-of', 'Gtiff', vrt_file , gtif_file]
    
    p = subprocess.Popen(args)
    
    
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(gtif_file, gdal.GA_ReadOnly)
    
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    
    print(str(ncol) + "cols and " + str(nrow) + "rows")
    
    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    
    raster_ds = None
    
    exampleRaster = output_path + '\\' +  'testExample.gtif'
    #The second example raster does nothing but prevent a runtime error.
    exampleRaster2 = output_path + '\\' +  'testExample2.gtif'
    
    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(exampleRaster, ncol, nrow, 1, gdal.GDT_Byte)
    
    out_raster_ds2 = memory_driver.Create(exampleRaster2, ncol, nrow, 1, gdal.GDT_Byte)
    
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    out_raster_ds.GetRasterBand(1).SetNoDataValue(-9999)
    b.Fill(0)
    
    
    
    
    
    # Rasterize the shapefile layer to our new dataset
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 layer,  # rasterize this layer
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [0],  # burn value 0
                                 ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                  'ATTRIBUTE=id']  # put raster values according to the 'id' field values
                                 )
    
    # Close dataset
    out_raster_ds = None
    out_raster_ds2 = None
    
    if status != 0:
        print("I don't think it worked...")
    else:
        print("Success")
    
    roi_ds = gdal.Open(exampleRaster, gdal.GA_ReadOnly)
    roi_ds.GetRasterBand(1).SetNoDataValue(-9999)
    roi = roi_ds.GetRasterBand(1).ReadAsArray()
    
    # How many pixels are in each class?
    classes = np.unique(roi)
    # Iterate over all class labels in the ROI image, printing out some information
    for c in classes:
        print('Class {c} contains {n} pixels'.format(c=c, n=(roi == c).sum()))
        
    ##########
    ############
    ##############
    ############
    ##########
    ############
    ########## Scikit-learn
    ############
    ##########
    ############
    ##############
    ############
    ##########
    ############
    ##############
    ############
    
    # Tell GDAL to throw Python exceptions, and register all drivers
    gdal.UseExceptions()
    gdal.AllRegister()
    
    # Read in our image and ROI image
    img_ds = gdal.Open(gtif_file, gdal.GA_ReadOnly)
    roi_ds = gdal.Open(exampleRaster, gdal.GA_ReadOnly)
    
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
        
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    
    # Display them
    plt.subplot(121)
    plt.imshow(img[:, :, 4], cmap=plt.cm.Greys_r)
    plt.title(dat_file[44:])
    
    plt.subplot(122)
    plt.imshow(roi, cmap=plt.cm.Spectral)
    plt.title('Gorse ROI Training Data')
    
    plt.show()
    
    
    #####
    
    
    # Find how many non-zero entries we have -- i.e. how many training data samples?
    n_samples = (roi > 0).sum()
    print('We have {n} samples'.format(n=n_samples))
    
    # What are our classification labels?
    labels = np.unique(roi[roi > 0])
    print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                    classes=labels))
    # We will need a "X" matrix containing our features, and a "y" array containing our labels
    #     These will have n_samples rows
    #     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster
    
    
    X = img[roi > 0, :]  # include 8th band, which is Fmask, for now
    y = roi[roi > 0]
    
    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y array is sized: {sz}'.format(sz=y.shape))
    
    #Clouds are already masked but we should make sure they're properly -9999
    clear = X[:, bandcount-2] > 0
    #clear = X[:, 6] > 0
    X[~clear] = -9999
    #X = X[clear, :7]  # we can ditch the Fmask band now
    y[~clear] = -9999
    ####
    
    
    #Random Forest
    
    # Initialize our model with 500 trees
    rf = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    # Fit our model to training data
    rf = rf.fit(X, y)
    
    
    #Gaussian Naive Bayes 
    
    gnb = GaussianNB()
    gnb = gnb.fit(X, y)
    
    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
    
    bands = list(range(1,bandcount-1)) #[1, 2, 3, 4, 5, 6]
    
    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
    
    ###
    
    
    # Setup a dataframe -- just like R
    df = pd.DataFrame()
    df['truth'] = y
    df['predict_rf'] = rf.predict(X)
    df['predict_gaus'] = gnb.predict(X)

    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict_rf'], margins=True))
    
    print(pd.crosstab(df['truth'], df['predict_gaus'], margins=True))

    # # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    
    img_as_array = img[:, :, :7].reshape(new_shape)
    #print(img_as_array)
    print('Reshaped from {o} to {n}'.format(o=img.shape,
                                            n=img_as_array.shape))
    
    # Now predict for each pixel
    class_prediction = rf.predict(img_as_array)
    
    print("The estimated accuracy of random forest is:" + str(rf.score(X,y)))
    
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    

    
    # Now predict for each pixel for gauss
    class_prediction_gauss = gnb.predict(img_as_array)
    
    # Reshape our classification map
    class_prediction_gauss = class_prediction_gauss.reshape(img[:, :, 0].shape)
    # # Visualize
    
    
        
    img543 = color_stretch(img, [4, 3, 2], (0, 8000))
    
    # See https://github.com/matplotlib/matplotlib/issues/844/
    n = class_prediction_gauss.max()
    # Next setup a colormap for our map
    colors = dict((
        (0, (0, 0, 0, 255)),  # Nodata
        (1, (250, 250, 0, 255)),  # Gorse
        (2, (20, 20, 100, 255)),  # Water
        (3, (255, 0, 0, 255)),  # Urban
        (4, (0, 150, 0, 255)),  # Agro
        (5, (0, 100, 0, 255)),  # Tree
        (6, (0, 40, 0, 255))  # Bog
    ))
    # Put 0 - 255 as float 0 - 1
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
        
    index_colors = [colors[key] if key in colors else 
                    (255, 255, 255, 0) for key in range(1, n + 1)]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)
    
    # Some example data to display
    # Now show the classmap next to the image
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(img543)
    
    ax2 = fig.add_subplot(222)
    ax2.imshow(class_prediction, cmap=cmap, interpolation='none')

    ax3 = fig.add_subplot(223)
    ax3.imshow(class_prediction_gauss, cmap=cmap, interpolation='none')

    
    ax1.title.set_text('Ground Truth')
    ax2.title.set_text('Random Forest')
    ax3.title.set_text('Naive Bayes')

    plt.tight_layout(2)
    plt.rcParams["figure.figsize"] = (8,5)
    plt.show()
    
    
    #Write out the raster where 1 is gorse presence and 0 is absence.

    
    rf_output = output_path + '\\' + 'RFPred' + dat_name[:-4] + '.gtif' 
    print(dat_name)
    print(rf_output)
    rf_save = memory_driver.Create(rf_output , ncol, nrow, 1, gdal.GDT_Byte)
    rf_save2 = memory_driver.Create(exampleRaster2 , ncol, nrow, 1, gdal.GDT_Byte)
    
    gorse_prediction = np.where(class_prediction == 1, 1, 0)

    
    print(gorse_prediction)

    out_gorse = rf_save.GetRasterBand(1)
    out_gorse.WriteArray(gorse_prediction)
    #Set the ROI image's projection and extent to our input raster's projection and extent
    rf_save.SetProjection(proj)
    rf_save.SetGeoTransform(ext)
    
    
    return class_prediction


for dat in LC8K07:

    print(dat)
    pred = image_analyzer(dat, gtif_file, layer)
    # compare = crop_and_assess(pred, gtruth, region, gtif_file)
    
for dat in LE7K07:
     # datF = r"C:\Users\Zam\Desktop\Masters\EOanalytics\SR" + "\\" + str(dat)
     pred = image_analyzer(dat, gtif_file, layer, bandcount = 7)
#     print(datF)
#     image_analyzer(datF, gtif_file, layer)
    

# gtif_J07 = r'C:\Users\Zam\Desktop\Masters\EOanalytics\test\testJ07.gtif'
# for dat in LC8J07:
#     datF = r"C:\Users\Zam\Desktop\Masters\EOanalytics\SR" + "\\" + str(dat)
#     print(datF)
#     image_analyzer(datF, gtif_J07, layerJ07)
    