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
    from wp4.constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN
    from wp4.baseline.spatial import get_spatial_baseline
except ImportError:
    from constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN
    from baseline.spatial import get_spatial_baseline


def get_spatiotemporal_baseline(
                         fe_lat: float,
                         fe_long: float,
                         timestamp: pd.Timestamp,
                         days: int,
                         pollutant: str,
                         years=[2015, 2016, 2017, 2018, 2019, 2020, 2021],
                         meteo_dataset: str = 'MERA',
                         min_distance_km: int = None,
                         max_distance_km: int = None,
                         upwind_downwind: str = None,
                         number_of_neighbours: int = 20,
                         mask_ocean: bool = False,
                         ) -> (pd.DataFrame, xr.Dataset, pd.DataFrame, pd.DataFrame):

    # For each year that the fire did not take place
    # load dataset
    pollutant_variable_name = POLLUTANTS[pollutant]['CAMS']
    ds_cams = xr.open_dataset(f"{DATA_DIR_CAMS_AN}/{pollutant_variable_name}.nc").copy()

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
    ds_fe_loc = ds_fe.sel(latitude=fe_lat, longitude=fe_long, method='nearest')

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
            days=days,
            hours = days*24,
            min_distance_km=min_distance_km,
            max_distance_km=max_distance_km,
            mask_ocean=mask_ocean,
            raw_data=True
            # loc_from_fe='upwind'
        )

        historical_baselines[year] = raw_data

    combined_data = None

    for key in historical_baselines:
        df_year = historical_baselines[key]
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

    # combine stats into a single dataframe
    df_baseline = pd.DataFrame({
        'time': df_index_time,
        'hour_from_event': df_hour_from_event,
        'spatiotemporal_baseline_mean': df_pixel_data_mean,
        'spatiotemporal_baseline_median': df_pixel_data_median,
        'spatiotemporal_baseline_lower_quartile': df_pixel_data_lower_quartile,
        'spatiotemporal_baseline_upper_quartile': df_pixel_data_upper_quartile,
        'spatiotemporal_baseline_std': df_pixel_data_std,
        'fire_event': list(ds_fe_loc.squeeze()[pollutant_variable_name].to_pandas())
    })

    return df_baseline


def test():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    query = """
                SELECT id, datetime, ST_X(geometry), ST_Y(geometry), source, location, reference, type, info
                FROM public.all_fire_events
                WHERE reference = 'Aqua' OR reference = 'Terra'
            """

    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})
    conn.close()

    for ind, fe in df_fire_events.head(1).iterrows():
        df_baseline = get_spatiotemporal_baseline(
            fe_lat=fe['latitude'],
            fe_long=fe['longitude'],
            timestamp=fe['datetime'],
            pollutant='PM25',
            days=2,
            min_distance_km=50,
            max_distance_km=500,
            mask_ocean=True
        )

        print(df_baseline)


if __name__ == '__main__':
    test()





