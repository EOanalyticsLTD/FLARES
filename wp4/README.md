
# FLARES WP4 - Fire emission detection and quantification from remote sensing  
  
## Dependencies  
  
The ***requirements.txt*** file lists the python packages needed to run the code in this repository. The easiest way to install all the packages at once is to use anaconda, as I found that installing cartopy using pip can be a hassle. To install all the dependencies run the following command: 
  
<code> conda install --file requirements.txt --channel conda-forge</code>   

If conda forge is not in your current channels, you can add it with

<code> conda config --add channels conda-forge </code>

## Configuration  
  
After installing the packages specified in the requirements.txt file, you need to enter some info in the configuration file ***config.cfg*** for the code to work properly.
You can find a short explanation for each of the sections of the configuration in the following section.

#### Data Directory - [DATA_DIR]
  
The path to the data directory, which will be the root directory to store all the data used in the scripts in. You can set the path of the directory where you want to store your data under the DATA_DIR section, with the dir_path key. 

#### Amazon Web Services - [AWS]

Holds the access and secret keys needed to access Mundi S3 buckets.

#### PostgreSQL/PostGIS Databse  - [DATABASE]

Holds the info needed to connect to the database on the Mundi Flares 2 instance. 

### APIS

Quite a few of the datasets used for WP4 are obtained via three different APIs:

* CAMS European Air Quality Analysis -> Atmosphere Data Store
* ERA5 reanalysis -> Climate Data Store 
* GFAS FRP & Wildfire Flux -> ECMWF

There are scripts running on the flares2 instance that automatically download and updata the data in the S3 buckets.   

#### Climate Data Store API - [CLIMATE_DATA_STORE]
Instructions on how to set up and use the api can be found [here](https://cds.climate.copernicus.eu/api-how-to). In the instructions they recommend to set up the .cdsapirc keyfile, however this gives problems when trying to download data from the Atmosphere Data Store, as it makes use of the same python package (called cdsapi) to make calls. Because of this, the credentials are entered when initiating the API client in the script. For this to work the key ***cds_key*** must be specified in the **CLIMATE_DATA_STORE** section of the configuration file. 
 
#### Atmosphere Data Store API - [ATMOSPHERE_DATA_STORE]

Instructions on how to set up and use the api can be found [here](https://ads.atmosphere.copernicus.eu/api-how-to). As mentioned before, the Atmosphere Data Store makes use of the same library for the API calls as the Climate Data Store, so the instructions are practically the same for both APIs the only difference being the key and url being used for the calls. So to be able to make calls to the ADS api the key ***ads_key*** must be specified in the  **ATMOSPHERE_DATA_STORE** section  configuration file. 

#### ECMWF API - [ECMWF_API]

Finally, the hourly FRP product and the wildfire flux daily averages can be downloaded via the ECMWF API. Instructions on how to set up the api can be found [here](https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api).  The key expires after a year and can be stored into a file or entered when initiating the ECMWF API client. For the scripts in this repo to work, please add your ***ecmwf_key*** and   
***ecmwf_email***  under the **ECMWF_API** section in the configuration file. 
  
## Data  
  
All the relevant datasets have been uploaded to S3 buckets and can be downloaded by running the following command in a terminal, with the working directory set to match the location of the flares_package.  
  
<code> python setup_data_folder.py -f -d</code>  
  
This command runs a script that sets up a folder structure on the data directory specified in the configuration file and  then downloads all the relevant datasets from the S3 buckets. Because the datasets are rather small compared to Landsat or Sentinel satellite data and  xarray handles the datasets quite efficiently, having the entire data collection on your local storage should be manageable. The total size of all the datasets combined is currently around 15GB.   

To check and download the latest version of the CAMS, ERA5 and GFAS datasets, you can run the following command.

<code> python setup_data_folder.py -u </code>

### Datasets 

##### Area of Interest
The datasets mentioned in this section have all been cropped to the following extent: 

##### CAMS European Air Quality Analysis   
I downloaded data for recent years from the three-year rolling archive in the [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-europe-air-quality-forecasts?tab=overview). Older data (2015-2017) was gathered from [this](http://www.regional.atmosphere.copernicus.eu/?&category=documentation&subensemble=macc_raq_resources) website. The datasets have been uplaoaded per pollutant to the ***cams-analysis*** S3 bucket. 

The following table provides an overview of the data available for each pollutant.    

|Year|NO|NO<sub>2</sub>|O<sub>3</sub>|CO|PM2.5|PM10 |SO<sub>2</sub>|PM10 Wildfires  
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:|  
| 2015 | no | yes | yes | no | yes | yes | no | no |  
| 2016 | no | yes | yes | yes | yes | yes |  yes |no |  
| 2017 | no | yes | yes | yes | yes | yes |  yes |no |  
| 2018 | yes | yes | yes | yes | yes | yes |  yes |no |  
| 2019 | yes | yes | yes | yes | yes | yes |  yes |no |  
| 2020 | yes | yes | yes | yes | yes | yes |  yes  |yes |  
| 2021 | yes | yes | yes | yes | yes | yes |  yes |yes |

##### fire mask

The cams-analysis bucket also contains a netcdf file with a mask for fire events (based on the fire events reported by the Terra & Aqua satellites, selected from the database table fire_events).
This mask can be used to remove any CAMS data possibly affected by other fire events when calculating the temporal baseline. If you update the database with more recent fire events, the mask can be regenerated using the notebook in the notebooks/baseline folder names create_fire_event_mask.   

##### MERA Analysis & Forecasts
Emily Gleeson provided us with Met Éireann ReAnalysis (MERA) data for U & V wind components, temperature and precipitation between 2015 and August 2019. More general information about the dataset can be found [here](https://www.met.ie/climate/available-data/mera). 

The original data came as compressed monthly grib1 files with a Lambert conformal conic grid projection. The data for the U & V components and the temperature is the analysis product, which is an 3hourly analysis output (00:00, 03:00,  06:00 etc.). For precipitation, however, there was no analysis data available for precipitation so we were given the 33 hour forecast product instead, which consists of the 1- to 33-hour forecasts from each 00 Z cycle. All the original compressed datasets have been uploaded to the ***mera-ingest*** S3 bucket.        

The final mera dataset that is used for deriving the spatial baseline was reprojected to a lat/long grid and resampled to match the CAMS 0.1x0.1 arcdegree resolution and converted to netcdf. For the processing, I used [CDO](https://code.mpimet.mpg.de/projects/cdo/) in a linux environment. 

To match the precipitation data with the rest of the analysis datasets, the first 24 hours of the 33 hour-forecasts were used for each day, and averaged for every three hours to match the temporal resolution of the other analysis products. The final results is the ***mera.nc*** which was compressed and uploaded to the ***mera-reanalysis*** S3 bucket.            
  
##### ERA5 Reanalysis

As the MERA dataset only goes up to August 2019 the [ERA5 Reanalysis](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) data was downloaded as a replacement dataset to use for periods when no MERA data is available. The ERA5 dataset set up contains the same data variables as the MERA , namely the U&V components, temperature & precipitation. There is a script running on the flares2 Mundi instance that updates the dataset with the latest data available twice a week. The latest version of the dataset can be found in the ***era5-reanalysis*** S3 bucket. 
  
##### GFAS Fire Radiative Product (FRP)  & Wildfire Flux 

The FRP is a gridded satellite data product (hourly), while the wildfire flux is a daily average, more information available [here](https://confluence.ecmwf.int/display/CKB/CAMS%3A+Global+Fire+Assimilation+System+%28GFAS%29+data+documentation). Wildfire flux data is available for the following pollutants:
 
* Carbon Dioxide  
* Carbon Monoxide  
* Methane  
* Nitrogen Oxides NOx  
* Particulate Matter PM2.5  
* Total Particulate Matter  
* Sulphur Dioxide

The FRP & Wildfire Flux datasets have been uploaded to the S3 bucket ***gfas***. 

### Database
A PostgreSQL/PostGIS database has been set up containing the following data:\

##### Air Quality Measurements 
Air Quality Measurements from various Ground Stations all over Ireland were provided by Stig. The data have been centralized per pollutant, each table containing ground measurements for the date & time per ground measurement station. Additionally, ground station information has been centralized in a the table ground_stations, including the geographical location of the ground station. 

##### Fire Events 
The data table ***fire_events*** contains all the fire events that were collected as part of WP2 and available on the Flares Drive. This table contains the following information for each fire event: 

* Unique fire ID (Primary Key)
* Date (and time if available) of the fire event report
* Location of the fire event as a geometry column
* Source of the report (Fire Brigade, Satellite or Media)
* Location name
* Reference - more detailed source of the event (which fire brigade, which satellite etc.)
* Type of fire (gorse, bog, bush, grass, forest or controlled fire) 
* Additional info & FRP,  if available   

## Flares Package  
I organised the code in the package in the following way: 
```
flares_package
│   constants.py          	# contains variables used throughout all scripts 
|   setup_data_folder.py  	# script to set up data folders and download data
│   config.cfg            	# configuration file
|   requirements.txt      	# dependencies
└───baseline
│   │   spatial.py        	# all the code for the spatial baseline derivation
│   │   temporal.py       	# all the code for the temporal baseline derivation
└───notebooks   
│   └───baseline          	# notebooks with example uses of the baseline functions
│   └───data_handling     	# API example download scripts, S3 download/upload scripts
│   └───validation_gr..         # the validation of ground station vs CAMS data
│   └───database          	# a few notebooks for the database fire events table creation
│   └───visualization     	# a few notebooks with data visualization
└───processing            	# processing scripts 
└───visualization         	# some visualisation stuff  
```

## Other handy tools  
  
##### Panoply Viewer  

Nice tool for quickly visualizing netcdf & grib files.  Can be downloaded from [this website](https://www.giss.nasa.gov/tools/panoply/download/)  
  
##### PGAdmin

Management tool for Postgres/PostGIS databases that allows you to get a quick overview of a postgres database, run queries & visualize geometries on a map. More info on the tool and how to download/install it [here](
https://www.pgadmin.org/download/).   
