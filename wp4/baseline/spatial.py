"""Spatial Baseline Derivation"""

# General Imports
import math
import warnings
from numpy.linalg import lstsq
from datetime import timedelta, datetime
from numpy import ones, vstack, round_, datetime64

# Database
import psycopg2, psycopg2.extras

# Data Handling
import metpy.calc
import pandas as pd
import xarray as xr

# Nearest Neighbour Selection
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# Local imports
try:
    from wp4.constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_ERA5, DATA_DIR_MERA, \
        DATA_DIR_CAMS_AN, EXTENTS
    from wp4.processing.helpers import create_dataset
except ImportError:
    from constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_ERA5, DATA_DIR_MERA, \
        DATA_DIR_CAMS_AN, EXTENTS
    from processing.helpers import create_dataset

warnings.simplefilter("ignore", RuntimeWarning)


# -- Functions for internal use


def _rotate(origin: tuple, point: tuple, angle: float) -> dict:
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    :param origin: tuple containing the coordinates (long, lat) of the origin
    :param point: tuple containing the coordinates (long, lat) of the point to rotate
    :param angle: the angle for the rotation, in radians.
    :return: dictionary containing the longitude and latitude of the rotated point/coordinate
    """

    o_long, o_lat = origin
    p_long, p_lat = point

    qx = o_long + math.cos(angle) * (p_long - o_long) - math.sin(angle) * (p_lat - o_lat)
    qy = o_lat + math.sin(angle) * (p_long - o_long) + math.cos(angle) * (p_lat - o_lat)

    return {'longitude': qx, 'latitude': qy}


def _rotate_coords(fe_coordinates: dict, wind_direction: float) -> dict:
    """
    Function that rotates a second coordinate around the location of the fire event based on the angle of the
    wind direction at the location of the fire event.

    :param fe_coordinates: coordinates of the fire event
    :param wind_direction: angle of the wind direction at the fire event (in degrees)
    :return: a dict containing the rotated coordinates
    """

    # create the dictionary of the second coordinates based on the original coordinates of the fire event.
    point = {
        'latitude': fe_coordinates['latitude'],
        'longitude': fe_coordinates['longitude'] + 1,
    }

    # the second coordinates are rotated based on the location of the fire event.
    rotated_point = _rotate(
        (fe_coordinates['longitude'], fe_coordinates['latitude']),
        (point['longitude'], point['latitude']),
        math.radians(-wind_direction)
    )

    return rotated_point  # return a dict containing the rotated coordinates


def _solve_lin_eq(coord_1: dict, coord_2: dict) -> (float, float):
    """
    Fits a line through two coordinates to obtain the function of the line separating upwind and downwind cells
    """

    points = [(coord_1['longitude'], coord_1['latitude']),
              (coord_2['longitude'], coord_2['latitude'])]  # extract the coordinate info
    x_coords, y_coords = zip(*points)  # group coordinates together based on direction (x/y)
    A = vstack([x_coords, ones(len(x_coords))]).T  # create array

    return lstsq(A, y_coords, rcond=None)[0]


def _label_upwind_cells(ds: xr.Dataset, wind_direction: float, coord_1: dict, coord_2: dict) -> pd.DataFrame:
    """
    Labels coordinate pair in the original dataset as upwind or downwind, based on the wind direction of the fire event

    :param ds: Dataset to which the boolean upwind/downwind array is added.
    :param wind_direction: wind direction at the fire event location.
    :param coord_1: Dictionary containing the coordinates of a point along a line crossing through the fire event, with
    the line perpendicular to the wind direction at the fire event. Dict contents: {'lat':latitude, 'long':longitude}
    :param coord_2:- Dictionary containing the second set coordinates of a point along a line crossing through the fire
    event, with the line perpendicular to the wind direction at the fire event. Dict contents: {'lat':latitude, 'long':longitude}
    :return:pandas dataframe, with the coordinates of each CAMS pixel and a boolean value to indicate if it the pixel is
    located upwind (True) or downwind (False).
    """
    # solve the lin equation to obtain the function of the line separating upwind and downwind cells
    m, c = _solve_lin_eq(coord_1, coord_2)

    def _n_or_s(direction):  # determine whether the wind comes from north or south
        if 90 <= direction <= 270:
            return 'N'  # North
        else:
            return 'S'  # South

    def _is_upwind(coord):  # calculates if a cams cell is located upwind or downwind from the fire event's location

        # calculate the max y/lat value for the coordinate using the parameters obtained from solving the lin. eq.
        max_y = (coord['longitude'] * m) + c

        # get wind direction (N or S)
        north_south = _n_or_s(wind_direction)

        # boolean check if cams cell latitude is over the max y/lat val
        bt_max_y = coord['latitude'] >= max_y

        # return True is upwind, False if downwind.
        if bt_max_y and north_south == 'N':
            return True
        elif (not bt_max_y) and north_south == 'S':
            return True
        else:
            return False

    df = ds.to_pandas().fillna(0)  # Convert the xarray dataset to a pandas dataframe and fill NA values with 0

    # Create and pandas dataframe where latitude will be the first column. The longitude vals will be used to name
    # the rest of the columns. This creates a raster, where the wind direction will be stored for each coordinate.
    df_result = pd.DataFrame({'latitude': df.index})

    for long in df.columns:  # loop over all the longitude values for the coordinates in the original dataset

        def _assign_label(cell):  # labels all upwind coordinates as True, downwind as False
            coord = {'longitude': long, 'latitude': cell['latitude']}

            return _is_upwind(coord)

        # the following line creates a dataframe to determine upwind/downwind for latitude in combination
        # with the long value
        df_single_col = df[long].reset_index().apply(_assign_label, 1)
        # assign the labeled values to a column in the result dataframe
        df_result[long] = df_single_col

    return df_result.set_index('latitude')


def _get_nearest_cams_data(coords: dict):
    """
    Extracts the info about the physical characteristics of the CAMS cell closest to the input coordinates

    :param coords: Dictionary containing coordinates - {'lat':latitude, 'long':longitude}.
    :return: Pandas dataframe containing info on the physical characteristics.
    """

    # Initiate connection to database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    # SQL Query to extract the desired information from the database
    nearest_query = f"""
            SELECT 
            *
            FROM public.spatial_baseline_info
            ORDER BY
            spatial_baseline_info.geometry <-> 'SRID=4326;POINT({coords['longitude']} {coords['latitude']})'::geometry
            LIMIT 1;
            """

    # Use pandas to store the result of the SQL query as a pandas dataframe.
    nearest_cams = pd.read_sql_query(nearest_query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})
    conn.close()

    return nearest_cams


def _get_physical_characteristics_data(coords: dict, EPSG: int = 4326) -> pd.DataFrame:
    """
    Collects the physical information for each CAMS pixel from the database, and calculates how far the cell is
    located from the fire event.

    :param coords: Dictionary containing coordinates - {'lat':latitude, 'long':longitude}.
    :param EPSG: integer corresponding to EPSG code to project the geometries to, defaults to 4326.
    :return: Pandas dataframe containing the physical characteristics for each CAMS cell to use for the NN search.
    """

    # Get coordinate information
    long = coords['longitude']
    lat = coords['latitude']

    # Initiate connection to the database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    # Set up the query to extract all the information per pixel coordinate from the database, leaving out no data pixels
    query = f"""
                SELECT
                id,
                ST_X(geometry),
                ST_Y(geometry),
                corine_grid_code,
                corine_label_3,
                elevation_avg,
                elevation_med,
                elevation_q1,
                elevation_q3,
                slope_avg,
                slope_med,
                slope_q1,
                slope_q3,
                ST_Distance(ST_SetSRID(ST_Point({round(long, 2)}, {round(lat, 2)}), {EPSG}), ST_SetSRID(geometry, {EPSG}), true) AS distance
                FROM public.spatial_baseline_info
                WHERE
                spatial_baseline_info.corine_grid_code NOT IN (128) 
            """

    # store the result as a pandas dataframe
    df = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})

    # close the connection to the database
    conn.close()

    # set a multiindex combining the latitude and longitude
    df = df.set_index(
        pd.MultiIndex.from_frame(df[['latitude', 'longitude']].round(2), names=('latitude', 'longitude'))
    )

    return df


def _combine_datasets(ds_features: xr.Dataset, df_lc_info: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the meteorological information and the land cover information into a single dataframe to use for the
    nearest neighbour search.

    :param ds_features: xarray dataset containing the meteorological information for each CAMS pixel during the time
    of the fire.
    :param df_lc_info: pandas dataframe, containing the physical characteristics for each of the CAMS pixel.
    :return: pandas dataframe with all the feature data for the nearest neighbour search.
    """
    df_features = ds_features.to_dataframe()  # convert the xarray dataset to a pandas dataframe
    df_features = df_features.rename(index=lambda val: round(val, 2))  # round the coordinates in the multiindex

    df_combined = df_lc_info.join(df_features)  # join the dataframes to create a dataframe containing all the info

    if 'quantile' in df_combined.columns:  # remove unwanted column from the dataframe
        df_combined = df_combined.drop(columns=['quantile'])

    return df_combined


def _create_feature_dataset(ds: xr.Dataset, features: list = None) -> xr.Dataset:
    """Creates an xArray dataset containing data to be used as features for selecting the most similar CAMS cells"""

    # check features param
    if features is None:
        features = ['t2m', 'tp', 'wind_speed']

    # create and empty dataset to store the data to be used as features for classification
    ds_features = xr.Dataset(coords={'longitude': ds.longitude, 'latitude': ds.latitude})

    # loop over features - t2m corresponds to temperature and tp total precipitation
    for feat in features:
        ds_features[f'25p_{feat}'] = ds[feat].quantile(0.25, dim='time')  # 25th percentile for each cell over time
        ds_features[f'75p_{feat}'] = ds[feat].quantile(0.75, dim='time')  # 75th percentile for each cell over time
        ds_features[f'mean_{feat}'] = ds[feat].mean(dim='time')  # mean for each cell over time
        ds_features[f'std_{feat}'] = ds[feat].std(dim='time')  # standard deviation for each cell over time

    # Wind direction is represented in degrees (value between 0 and 360), so simply taking the average wind direction
    # will not give a meaningful result. Instead the u and v components are averaged and used to calculate the average
    # wind direction.

    mean_u = xr.DataArray(ds['u10'].mean(dim='time'), attrs=ds['u10'].attrs)
    mean_v = xr.DataArray(ds['v10'].mean(dim='time'), attrs=ds['v10'].attrs)

    ds_features['mean_wind_direction'] = metpy.calc.wind_direction(mean_u, mean_v, convention="to")

    return ds_features


def _get_upwind_cells(ds_wind_direction: xr.Dataset, fe_timestamp: pd.Timestamp, fe_coords: dict) -> xr.DataArray:
    """
    Function that creates a boolean raster where CAMS cells upwind of a fire event are labeled True and CAMS cells
    downwind of a fire event are labeled False.

    :param ds_wind_direction: xarray dataset containing the wind direction information.
    :param fe_timestamp: timestamp indicating the date and time the fire event was reported.
    :param fe_coords: dictionary containing the lat/long coordinates of the reported fire event.
    :return: xarray boolean DataArray where CAMS cells upwind of a fire event are labeled True and CAMS cells
    downwind of a fire event are labeled False.
    """

    # get the wind direction at the time and location of the fire event
    ds_fe_wind_direction_time = ds_wind_direction.sel(
        time=fe_timestamp.replace(minute=0),
        method='nearest'
    )

    ds_fe_wind_direction_loc = ds_fe_wind_direction_time.sel(
        latitude=fe_coords['latitude'],
        longitude=fe_coords['longitude'],
        method='nearest'
    )

    # convert the the direction as float
    angle_fe_wind_direction = float(ds_fe_wind_direction_loc['wind_direction'].values)

    # generate a second set of coordinates
    rotated_coord = _rotate_coords(fe_coords, angle_fe_wind_direction)

    # label CAMS pixels as upwind/downwind based on the location of the fire event and the wind direction at that time
    df_upwind = _label_upwind_cells(
        ds_fe_wind_direction_time['wind_direction'],
        angle_fe_wind_direction,
        fe_coords,
        rotated_coord,
    )

    # convert pandas object to numpy and store as Xarray dataarray.
    ar_upwind = xr.DataArray(
        df_upwind.to_numpy(),
        [("latitude", list(df_upwind.index)), ("longitude", list(df_upwind.columns))]
    )

    return ar_upwind


def calculate_wind_speed_direction(ds: xr.Dataset, u_var: str = 'u10', v_var: str = 'v10',
                                   convention: str = "to") -> xr.Dataset:
    """
    Calculates the wind speed and direction for an xArray dataset containing both u and v data.

    To calculate the wind direction & speed, the metpy library is used.
    For more info, see the documentation here: https://unidata.github.io/MetPy/latest/index.html

    :param ds: xArray dataset containing U and V data.
    :param u_var: string, name of the U data variable in the dataset, defaults to 'u10'.
    :param v_var: string, name of the V data variable in the dataset, defaults to 'v10'.
    :param convention: string, from metpy docs: Convention to return direction; ‘from’ returns the direction the wind
    is coming from (meteorological convention), ‘to’ returns the direction the wind is going towards (oceanographic
    convention). Default is set to: 'to'.
    :return: original xArray dataset with wind speed and direction added as additional data variables.
    """

    # Calculate wind speed and wind direction and add the data to the input dataset
    ds['wind_direction'] = metpy.calc.wind_direction(ds[u_var], ds[v_var], convention=convention)
    ds['wind_speed'] = metpy.calc.wind_speed(ds[u_var], ds[v_var])

    return ds


def _normalise_data(df: pd.DataFrame) -> (pd.DataFrame, preprocessing.MinMaxScaler):
    """
    Normalises the feature data in the dataframe using the scitkit-learn MinMaxScaler:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    :param df: Pandas dataframe containing the feature data for the training of the classifier/
    :return: pandas dataframe, containing the data normalised to values between 0-1 & the scikit-learn MinMaxScaler obj.
    """
    df_array = df.values  # get the dataframe as a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    df_array_scaled = min_max_scaler.fit_transform(df_array)  # scale the data
    # Return the scaled data as a pandas dataframe, using the same index and column information as the input dataframe
    # Also returns the MinMaxScaler object
    return pd.DataFrame(df_array_scaled, columns=df.columns, index=df.index), min_max_scaler


def _filter_dataset(df: pd.DataFrame, min_distance: int, max_distance: int,
                    loc_from_fe: str, mask_ocean: bool) -> pd.DataFrame:
    """
    Filter which CAMS pixels to use for the nearest neighbour search based on the parameters set by the user

    :param df: pandas dataframe containing the feature data
    :param min_distance: minimum distance (in km) from the fire event
    :param max_distance: maximum distance (in km) from the fire event
    :param loc_from_fe: specify whether to only use pixels upwind or downwind from the fire location. 'upwind', ''
    :param mask_ocean: whether to mask ocean pixels or not
    :return:"""

    if min_distance is not None:
        min_distance_m = min_distance * 1000
        df = df[df['distance'] >= min_distance_m]

    if max_distance is not None:
        max_distance_m = max_distance * 1000
        df = df[df['distance'] <= max_distance_m]

    if mask_ocean:
        df = df[df['corine_grid_code'] != 44]

    if loc_from_fe == 'upwind':
        df = df[df['upwind'] == True]
    elif loc_from_fe == 'downwind':
        df = df[df['upwind'] == False]

    return df


def _open_meteo_dataset(fe_timestamp: pd.Timestamp, hours: int, meteo_dataset: str) -> xr.Dataset:
    """
    Loads the meteorological dataset as an Xarray Dataset for the specified date and time window

    :param meteo_dataset: string indicating which meteorological dataset to use - 'ERA5' for the era5 dataset and 'MERA'
    for the Met Éireann dataset.
    :param fe_timestamp: timestamp indicating the date and time the fire event was reported
    :param hours: Number of hours to use as a time window before and after the fire event for selecting meteorological
    data to determine the CAMS cells most similar to the cell closest to the fire event.
    :return: The meteorological dataset as an xArray Dataset
    """

    # load the meteorological dataset
    if meteo_dataset == 'ERA5':
        ds_meteo = xr.open_dataset(f'{DATA_DIR_ERA5}/era5.nc')
        ds_meteo = ds_meteo.sel(
            time=slice(
                fe_timestamp.replace(minute=0) - timedelta(hours=hours),
                fe_timestamp.replace(minute=0) + timedelta(hours=hours)
            )
        )

        return ds_meteo
    elif meteo_dataset == 'MERA':
        ds_meteo = xr.open_dataset(f'{DATA_DIR_MERA}/mera.nc')
        ds_meteo = ds_meteo.sel(
            time=slice(
                fe_timestamp.replace(minute=0) - timedelta(hours=hours),
                fe_timestamp.replace(minute=0) + timedelta(hours=hours)
            )
        )

        ds_meteo['longitude'] = round_(ds_meteo.longitude.values, 2)

        # resample xarray to match CAMS grid -- possible solution https://xesmf.readthedocs.io/en/latest/
        # Maybe better to just resample the entire dataset once, instead of every run

        return ds_meteo

    else:  # for any other value raise an error
        raise ValueError(f"Entered dataset '{meteo_dataset}' does not exist - Please select either 'ERA5' or 'MERA'")


# -- Main functions of this module


def get_data_nn(fe_timestamp: pd.Timestamp, fe_coords: dict, meteo_dataset: str = 'MERA', hours: int = 24) -> (
        pd.DataFrame, pd.DataFrame):
    """
    Function that combines all data gathering steps and sets up the pandas DataFrame that is used for the Nearest
    Neighbour (NN) search.

    :param fe_timestamp: timestamp indicating the date and time the fire event was reported.
    :param fe_coords: dictionary containing the lat/long coordinates of the reported fire event.
    :param meteo_dataset: string indicating which meteorological dataset to use - 'ERA5' for the era5 dataset and 'MERA'
    for the Met Éireann dataset.
    :param hours: Number of hours to use as a time window before and after the fire event for selecting meteorological
    data to determine the CAMS cells most similar to the cell closest to the fire event. Defaults to 24.
    :return: Returns a dataframe containing the feature data for the nearest neighbour search and a dataframe containing
    the feature information for the CAMS pixel closest to the location of the fire event.
    """

    if (fe_timestamp >= datetime(year=2019, month=8, day=30)) and (meteo_dataset == 'MERA'):
        # print('No MERA data available after 31-08-2019, using ERA5 dataset instead')
        meteo_dataset = 'ERA5'

    # Load the meteorological dataset for the specified time period
    ds_meteo = _open_meteo_dataset(fe_timestamp, hours, meteo_dataset)

    # calculate and wind speed/direction to the dataset with meteorological data
    ds_meteo = calculate_wind_speed_direction(ds_meteo)

    # create a dataset with feature data for the nearest neighbour selection
    ds_features = _create_feature_dataset(ds_meteo)

    # label upwind/downwind cells
    ar_upwind = _get_upwind_cells(ds_meteo, fe_timestamp, fe_coords)

    ds_features['upwind'] = ar_upwind

    # Get land cover, elevation, slope info from database
    df_lc = _get_physical_characteristics_data(fe_coords)

    # Combine feature dataset and the land cover information dataframe into a single dataframe
    df_classification = _combine_datasets(ds_features, df_lc)

    # split up dataframe, separating the row containing the info of the CAMS cell closest to the fire event
    df_closest_cell = _get_nearest_cams_data(fe_coords)

    df_fe_information = df_classification[df_classification['id'] == df_closest_cell['id'].iloc[0]]
    df_classification = df_classification[df_classification['id'] != df_closest_cell['id'].iloc[0]]

    return df_classification, df_fe_information


def perform_nn_search(df_fe: pd.DataFrame, df_clf: pd.DataFrame, columns: list, number_of_neighbours: int,
                      algorithm: str = 'ball_tree') -> pd.DataFrame:
    """
    Function that performs the nearest neighbour search

    :param df_fe: pandas dataframe containing the physical and meteorological information of the CAMS pixel closest to
    the fire event.
    :param df_clf: pandas dataframe containing the physical and meteorological information of the CAMS pixels over
    Ireland, excluding the pixel closest to the fire event.
    :param columns: list of column names to use for the nearest neighbour search.
    :param number_of_neighbours: number of neighbours to find
    :param algorithm: Algorithm used to compute the nearest neighbours, more information here:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    :return: pandas dataframe containing the information of the nearest neighbours
    """

    # reset the index of the dataframe, as the NN result gives the index of the nearest neighbours
    df_clf = df_clf.reset_index()
    # select the specified columns in the dataframe
    df_data = df_clf[columns]

    # normalise the data in the dataframe
    df_data, min_max_scaler = _normalise_data(df_data)

    # normalise the data of the pixel closest to the location of the fire event. This will be the pixel for which the
    # algorithm will return the nearest neighbours
    df_fe_scaled = min_max_scaler.transform(df_fe[columns].values)
    df_closest_pixel_fe = pd.DataFrame(df_fe_scaled, columns=columns)

    # train the NN searcher
    nn_searcher = NearestNeighbors(n_neighbors=number_of_neighbours, algorithm=algorithm).fit(df_data)

    # perform the NN search, returns the distance in feature space and the index in the dataframe (df_clf)
    distances, indices = nn_searcher.kneighbors(df_closest_pixel_fe)
    # select the rows in the classification dataframe belonding to the nearest neighbours based on the indices returned
    # by the NN searcher
    df_result = df_clf[df_clf.index.isin(list(indices[0]))]

    return df_result


def get_spatial_baseline(fe_lat: float,
                         fe_long: float,
                         timestamp: pd.Timestamp,
                         pollutant: str,
                         days: int,
                         hours: int = 24,
                         meteo_dataset: str = 'MERA',
                         min_distance_km: int = None,
                         max_distance_km: int = None,
                         upwind_downwind: str = None,
                         number_of_neighbours: int = 50,
                         mask_ocean: bool = False,
                         raw_data: bool = False,
                         extent='IRELAND',
                         ) -> (pd.DataFrame, xr.Dataset, pd.DataFrame, pd.DataFrame):
    """
    Main function to calculate the spatial baseline, and get all the information related to the spatial baseline.

    :param fe_lat: latitude of the fire event.
    :param fe_long: longitude of the fire event.
    :param timestamp: timestamp of the fire event.
    :param pollutant: string indicating the pollutant for which to calculate the baseline for. Options are: 'CO','O3',
    'NO','NO'2,'PM'25,'PM'10,'SO2'.
    :param days: number of days to use for the time window around the fire event.
    :param hours: Number of hours to use as a time window before and after the fire event for selecting meteorological
    data to determine the CAMS cells most similar to the cell closest to the fire event. Defaults to 24.
    :param meteo_dataset: string indicating which meteorological dataset to use - 'ERA5' for the era5 dataset and 'MERA'
    for the Met Éireann dataset. Default dataset is 'MERA', if not data available it switches to ERA5 automatically.
    :param min_distance_km: minimum distance (in km) from the fire event
    :param max_distance_km: maximum distance (in km) from the fire event
    :param upwind_downwind: specify whether to only use pixels upwind or downwind from the fire location. Options are:
    'upwind', 'downwind' or the default: None.
    :param number_of_neighbours: number of neighbours you want the Nearest Neighbour (NN) algorithm to find.
    :param mask_ocean: whether to mask ocean pixels for the NN search or not
    :param raw_data: Flag, when True return the dataframe containing all the pollutant concentration values per pixel
    :param extent: string of the key of the extent to use in the EXTENTS dictionary, defaults to IRELAND.
    :return:
    df_baseline: pandas dataframe containing all the baseline information,
    ds_fe: xArray dataset containing all the feature information generated from the meteo data for the NN search,
    nn_result: pandas dataframe containing the result from the NN search and all the information for each match,
    df_fe_information: pandas dataframe containing all the information of the target pixel of the NN search

    if raw_data is True:
    df_pixel_data: Pandas dataframe containing the pollutant concentration per pixel for all the selected pixels.
    """

    if not EXTENTS[extent]['WEST'] <= fe_long <= EXTENTS[extent]['EAST']:
        #    f'Longitude: {fe_long} is outside of the study area extent: {EXTENTS[extent]["WEST"]} - {EXTENTS[extent]["EAST"]}')
        return None

    if not EXTENTS[extent]['SOUTH'] <= fe_lat <= EXTENTS[extent]['NORTH']:
        #    f'Latitude: {fe_lat} is outside of the study area extent: {EXTENTS[extent]["SOUTH"]} - {EXTENTS[extent]["NORTH"]}')
        return None

    # Next, derive the spatial baseline for the pixels returned by the nearest neighbour search
    # Load the CAMS dataset
    pollutant_variable_name = POLLUTANTS[pollutant]['CAMS']

    years = list({(timestamp.replace(minute=0) - timedelta(days=days)).year,
                  (timestamp.replace(minute=0) + timedelta(days=days)).year})

    ds_cams = create_dataset(pollutant, years=years)

    if ds_cams is None:
        # print(f'No CAMS data available for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}.')
        if raw_data:
            return None
        else:
            return None, None, None, None

    # Initiate a dictionary containing the coordinates of the fire event
    fe_coords = {
        'longitude': fe_long,
        'latitude': fe_lat,
    }

    # Get the dataframes for the nearest neighbour search
    df_classification, df_fe_information = get_data_nn(
        timestamp,
        fe_coords,
        hours=hours,
        meteo_dataset=meteo_dataset
    )

    df_classification = df_classification.drop(columns=['longitude', 'latitude']).dropna().reset_index()

    # filter out unwanted rows based on the parameters set by the user
    df_classification = _filter_dataset(df_classification, min_distance_km, max_distance_km, upwind_downwind,
                                        mask_ocean)

    # specify the names of the columns containing the data we want to base the nearest neighbour search on
    columns = [
        'corine_grid_code',
        'elevation_avg',
        'elevation_med',
        'elevation_q1',
        'elevation_q3',
        'slope_avg',
        'slope_med',
        'slope_q1',
        'slope_q3',
        '25p_t2m',
        '75p_t2m',
        'mean_t2m',
        'std_t2m',
        '25p_tp',
        '75p_tp',
        'mean_tp',
        'std_tp',
        '25p_wind_speed',
        '75p_wind_speed',
        'mean_wind_speed',
        'std_wind_speed',
        'mean_wind_direction'
    ]

    # perform the nearest neighbour search
    nn_result = perform_nn_search(
        df_fe_information,
        df_classification,
        columns,
        number_of_neighbours=number_of_neighbours
    )

    # Select CAMS data for the period of observation
    ds_fe = ds_cams.sel(time=slice(
        timestamp.replace(minute=0) - timedelta(days=days),
        timestamp.replace(minute=0) + timedelta(days=days)
    ))

    # Create a dataset containing the CAMS data closest to the location of the fire
    ds_fe_loc = ds_fe.sel(longitude=fe_long, latitude=fe_lat, method='nearest')

    # If no data return None
    if ds_fe.time.size == 0:
        # print(f'No CAMS data available for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}.')
        return None, None, None, None

    # Now iterate over the pixels returned by the nearest neighbour search and store the data in a pandas df
    pixel_data_l = []

    for ind, pixel in nn_result.iterrows():
        # select the data for the pixel
        pixel_data = ds_fe.sel(latitude=pixel['latitude'], longitude=pixel['longitude'], method='nearest')

        df_pixel_data = pixel_data.to_dataframe()[[pollutant_variable_name]]  # convert the dataset to a pandas df

        # in case there is no data for this pixel, move on with the next pixel
        if len(df_pixel_data.dropna()) == 0:
            continue

        df_pixel_data = df_pixel_data.rename(  # assign a unique column name based on the index
            columns={pollutant_variable_name: f'{pollutant_variable_name}_{ind}'}
        )

        pixel_data_l += [df_pixel_data]  # add to the list of dataframes

    df_pixel_data = pd.concat(pixel_data_l, axis=1)  # concatenate the list of dataframes into a single dataframe

    # calculate the statistics representative of the baseline concentration range
    df_index = df_pixel_data.index
    df_pixel_data_mean = df_pixel_data.mean(axis=1)
    df_pixel_data_median = df_pixel_data.median(axis=1)
    df_pixel_data_lower_quartile = df_pixel_data.quantile(0.25, axis=1)
    df_pixel_data_upper_quartile = df_pixel_data.quantile(0.75, axis=1)
    df_pixel_data_std = df_pixel_data.std(axis=1)

    def hour_from_event(x):
        """Function to calculate hour difference from a fire event"""

        x_time = x.values[0]  # select the timestamp

        if type(x_time) == datetime64:  # if the datetime is a numpy datetime64 object convert to datetime object
            x_time = pd.Timestamp(x_time)

        hours = x_time - timestamp.replace(minute=0)  # difference between the hour and the fire event timestamp
        return int(hours.total_seconds() / 3600)  # calculate the number of hours

    # Creates a hourly datetime range to use as index
    df_index_time = pd.date_range(
        timestamp.replace(minute=0) - timedelta(days=days),
        timestamp.replace(minute=0) + timedelta(days=days),
        freq='h'
    ).tolist()

    # Calculates how far from fire event each datetime object occurs
    df_hour_from_event = pd.DataFrame({'time': df_index_time}).apply(hour_from_event, 1)

    if df_hour_from_event.index.size != df_index.size:
        # print(f'Missing CAMS data for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}. Baseline calculation stopped.')
        if raw_data:
            return None
        else:
            return None, None, None, None

    df_hour_from_event.index = df_index

    if raw_data:
        return df_pixel_data
    else:

        # combine stats into a single dataframe
        df_baseline = pd.DataFrame({
            'time': df_index,
            'hour_from_event': df_hour_from_event,
            'spatial_baseline_mean': df_pixel_data_mean,
            'spatial_baseline_median': df_pixel_data_median,
            'spatial_baseline_lower_quartile': df_pixel_data_lower_quartile,
            'spatial_baseline_upper_quartile': df_pixel_data_upper_quartile,
            'spatial_baseline_std': df_pixel_data_std,
            'fire_event': list(ds_fe_loc.squeeze()[pollutant_variable_name].to_pandas())
        })

        return df_baseline, ds_fe, nn_result, df_fe_information


def test_baseline():
    """Function to test spatial baseline, runs when spatial.py is run as the main script"""
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    query = """
            SELECT id, datetime, ST_X(geometry), ST_Y(geometry), source, location, reference, type, info
            FROM public.all_fire_events
            WHERE reference = 'Aqua' OR reference = 'Terra'
        """

    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})
    conn.close()

    for ind, fe in df_fire_events[150:200].iterrows():

        baseline_df, ds_fe, nn_result, df_fe_information = get_spatial_baseline(
            fe_lat=fe['latitude'],
            fe_long=fe['longitude'],
            timestamp=fe['datetime'],
            pollutant='NO',
            days=2,
            min_distance_km=50,
            max_distance_km=500,
            mask_ocean=True
            # loc_from_fe='upwind'
        )

        print(baseline_df)


if __name__ == '__main__':
    test_baseline()
