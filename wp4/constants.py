import os
import configparser
import pathlib

# PROJECT BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the configuration
config = configparser.ConfigParser(interpolation=None)
config.read(os.path.join(BASE_DIR, 'config.cfg'))

# S3
S3_ACCESS_KEY = config['AWS']['s3_access_key']
S3_SECRET_KEY = config['AWS']['s3_secret_key']
S3_ENDPOINT = config['AWS']['s3_endpoint']
S3_REGION_NAME = config['AWS']['s3_region_name']

# APIS

# Climate Data Store
CDS_URL = config['CLIMATE_DATA_STORE']['cds_url']
CDS_KEY = config['CLIMATE_DATA_STORE']['cds_key']

# Atmosphere Data Store
ADS_URL = config['ATMOSPHERE_DATA_STORE']['ads_url']
ADS_KEY = config['ATMOSPHERE_DATA_STORE']['ads_key']

# ECMWF API
ECMWF_URL = config['ECMWF_API']['ecmwf_url']
ECMWF_KEY = config['ECMWF_API']['ecmwf_key']
ECMWF_EMAIL = config['ECMWF_API']['ecmwf_email']

# DATABASE
DB_HOST = config['DATABASE']['host']
DB_NAME = config['DATABASE']['name']
DB_USER = config['DATABASE']['user']
DB_PASS = config['DATABASE']['password']

# DATA DIRECTORY
DATA_DIR = pathlib.Path(config['DATA_DIR']['dir_path'])

# DATASET DIRECTORIES
DATA_DIR_CAMS_AN = DATA_DIR.joinpath("cams/").as_posix()
DATA_DIR_CAMS_RE = DATA_DIR.joinpath("cams_reanalyses/").as_posix()
DATA_DIR_ERA5 = DATA_DIR.joinpath("era5").as_posix()
DATA_DIR_MERA = DATA_DIR.joinpath("mera/").as_posix()
DATA_DIR_GFAS = DATA_DIR.joinpath("gfas/").as_posix()
DATA_DIR_DEM = DATA_DIR.joinpath("dem/").as_posix()
DATA_DIR_LC = DATA_DIR.joinpath("land_cover/").as_posix()
DATA_DIR_FIRES = DATA_DIR.joinpath("fire_events/").as_posix()

# PLOTS, FIGURES a& OTHER OUPUTS
DATA_DIR_PLOTS = DATA_DIR.joinpath("plots/").as_posix()


# CAMS POLLUTANT INFO
CO = {
    'CAMS':'co_conc',
    'ADS_PARAM':'carbon_monoxide',
    'FULL_NAME':'Carbon Monoxide',
    'FORMULA':'CO',
    'FORMULA_HTML':'CO',

}

O3 = {
    'CAMS':'o3_conc',
    'ADS_PARAM':'ozone',
    'FULL_NAME':'Ozone',
    'FORMULA':'O3',
    'FORMULA_HTML':'O<sub>3</sub>',
}

NO = {
    'CAMS':'no_conc',
    'ADS_PARAM':'nitrogen_monoxide',
    'FULL_NAME':'Nitrogen Oxide',
    'FORMULA':'NO',
    'FORMULA_HTML':'NO',
}

NO2 = {
    'CAMS':'no2_conc',
    'ADS_PARAM':'nitrogen_dioxide',
    'FULL_NAME':'Nitrogen Dioxide',
    'FORMULA':'NO2',
    'FORMULA_HTML':'NO<sub>2</sub>',
}

PM25 = {
    'CAMS':'pm2p5_conc',
    'ADS_PARAM':'particulate_matter_2.5um',
    'FULL_NAME':'Fine Particulate Matter',
    'FORMULA':'PM2.5',
    'FORMULA_HTML':'PM2.5',
}

PM10 = {
    'CAMS':'pm10_conc',
    'ADS_PARAM':'particulate_matter_10um',
    'FULL_NAME':'Coarse Particulate Matter',
    'FORMULA':'PM10',
    'FORMULA_HTML':'PM10',
}

SO2 = {
    'CAMS':'so2_conc',
    'ADS_PARAM':'sulphur_dioxide',
    'FULL_NAME':'Sulphur Dioxide',
    'FORMULA':'SO2',
    'FORMULA_HTML':'SO<sub>2</sub>',
}

POLLUTANTS = {
    'CO':CO,
    'O3':O3,
    'NO':NO,
    'NO2':NO2,
    'PM25':PM25,
    'PM10':PM10,
    'SO2':SO2,
}

EXTENTS = {  # extent of the area of interest, format -> [N,W,S,E]
    'IRELAND':{
        'LIST':[55.65, -11.35, 51.35,-5.25],
        'NORTH':55.65,
        'WEST':-11.35,
        'SOUTH':51.35,
        'EAST':-5.25,
    }
}
