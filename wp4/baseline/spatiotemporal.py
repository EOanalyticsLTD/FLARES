"""Baseline combining temporal & spatial factors"""

# General Imports
from datetime import timedelta
from numpy import datetime64


# Database
import psycopg2, psycopg2.extras

# Data Handling
import pandas as pd
import xarray as xr

# Local imports
try:
    from wp4.constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, EXTENTS
    from wp4.baseline.spatial import get_spatial_baseline
    from wp4.processing.helpers import create_dataset
except ImportError:
    from constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, EXTENTS
    from baseline.spatial import get_spatial_baseline
    from processing.helpers import create_dataset


def get_spatiotemporal_baseline(
        fe_lat: float,
        fe_long: float,
        timestamp: pd.Timestamp,
        days: int,
        pollutant: str,
        years=None,
        meteo_dataset: str = 'MERA',
        min_distance_km: int = None,
        max_distance_km: int = None,
        number_of_neighbours: int = 20,
        mask_ocean: bool = False,
        extent='IRELAND',) -> (pd.DataFrame, xr.Dataset, pd.DataFrame, pd.DataFrame):

    if not EXTENTS[extent]['WEST'] <= fe_long <= EXTENTS[extent]['EAST']:
        print(
            f'Longitude: {fe_long} is outside of the study area extent: {EXTENTS[extent]["WEST"]} - {EXTENTS[extent]["EAST"]}')
        return None

    if not EXTENTS[extent]['SOUTH'] <= fe_lat <= EXTENTS[extent]['NORTH']:
        print(
            f'Latitude: {fe_lat} is outside of the study area extent: {EXTENTS[extent]["SOUTH"]} - {EXTENTS[extent]["NORTH"]}')
        return None
    # check years param
    if years is None:
        years=[2015, 2016, 2017, 2018, 2019, 2020, 2021]

    # For each year that the fire did not take place
    # load dataset
    pollutant_variable_name = POLLUTANTS[pollutant]['CAMS']
    ds_cams = create_dataset(pollutant, years=years).copy()[pollutant_variable_name]

    if pd.to_datetime(timestamp.round('h')) not in ds_cams.time:
        print(f'No CAMS data available for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}.')
        return None

    fe_year = timestamp.year  # Get the year that fire event took place

    # Select CAMS data for the period of observation
    ds_fe = ds_cams.sel(time=slice(
        timestamp.replace(minute=0) - timedelta(days=days),
        timestamp.replace(minute=0) + timedelta(days=days)
    ))

    # If no data return None
    if ds_fe.time.size == 0:
        return None

    # Select data from the nearest CAMS cell to the fire event
    ds_fe = ds_fe.sel(latitude=fe_lat, longitude=fe_long, method='nearest')

    # # mask data for other fire event that occured during the period of time used for the FE analysis
    # if fire_mask:
    #     ds_cams = _apply_fe_mask(ds_cams)

    historical_baselines = {}

    for year in [y for y in years if y != fe_year]:
        if not ((timestamp.day == 29) and (timestamp.month == 2)):
            prev_timestamp = timestamp.replace(minute=0, year=year)
        else:
            prev_timestamp = timestamp.replace(minute=0, day=1, month=3, year=year)

        raw_data = get_spatial_baseline(
            fe_lat= fe_lat,
            fe_long= fe_long,
            timestamp= prev_timestamp,
            pollutant= pollutant,
            days= days,
            hours = days*24,
            min_distance_km=min_distance_km,
            max_distance_km=max_distance_km,
            mask_ocean=mask_ocean,
            raw_data=True,
            meteo_dataset=meteo_dataset,
            number_of_neighbours=number_of_neighbours,
        )

        if raw_data is not None:
            historical_baselines[year] = raw_data

    combined_data = None

    for key, df_year in historical_baselines.items():
        new_cols = {x:f'{key}_{x}' for x in df_year.columns}
        df_year = df_year.rename(columns=new_cols)
        df_year = df_year.reset_index().drop(columns=['time'])

        if combined_data is None:
            combined_data = df_year
        else:
            combined_data = pd.concat([combined_data, df_year], axis=1)

    # calculate the statistics representative of the baseline concentration range
    df_index = combined_data.index
    df_pixel_data_mean = combined_data.mean(axis=1)
    df_pixel_data_median = combined_data.median(axis=1)
    df_pixel_data_lower_quartile = combined_data.quantile(0.25, axis=1)
    df_pixel_data_upper_quartile = combined_data.quantile(0.75, axis=1)
    df_pixel_data_std = combined_data.std(axis=1)

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
    df_hour_from_event.index = df_index

    ar_fire_event = list(ds_fe.squeeze().to_pandas())

    # in case of missing data, fill the missing hours with None values
    if df_hour_from_event.index.size != len(ar_fire_event):

        # combined dataframe based on datetime index, missing dates will have NA col values
        df_fill = pd.DataFrame(index=df_index_time)
        df_fe = pd.DataFrame(ds_fe.squeeze().to_pandas(), columns=[ds_fe.name], index=ds_fe.time)
        df_filled = df_fill.join(df_fe)

        # convert to list
        ar_fire_event = df_filled[ds_fe.name].tolist()

    # combine stats into a single dataframe
    df_baseline = pd.DataFrame({
        'time': df_index_time,
        'hour_from_event': df_hour_from_event,
        'spatiotemporal_baseline_mean': df_pixel_data_mean,
        'spatiotemporal_baseline_median': df_pixel_data_median,
        'spatiotemporal_baseline_lower_quartile': df_pixel_data_lower_quartile,
        'spatiotemporal_baseline_upper_quartile': df_pixel_data_upper_quartile,
        'spatiotemporal_baseline_std': df_pixel_data_std,
        'fire_event': ar_fire_event
    })

    return df_baseline


def test_baseline():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    query = """
                SELECT id, datetime, ST_X(geometry), ST_Y(geometry), source, location, reference, type, info
                FROM public.all_fire_events
                WHERE reference = 'Aqua' OR reference = 'Terra'
            """

    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})
    conn.close()

    for ind, fe in df_fire_events[150:155].iterrows():
        print(ind)
        df_baseline = get_spatiotemporal_baseline(
            fe_lat=fe['latitude'],
            fe_long=fe['longitude'],
            timestamp=fe['datetime'],
            pollutant='NO',
            days=2,
            min_distance_km=50,
            max_distance_km=500,
            mask_ocean=True
        )

        print(df_baseline)


if __name__ == '__main__':
    test_baseline()





