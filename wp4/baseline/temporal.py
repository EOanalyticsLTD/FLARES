"""Temporal Baseline Derivation"""

import calendar
import psycopg2
import pandas as pd
import xarray as xr
from pathlib import Path
from numpy import datetime64
from datetime import timedelta

# Local imports
try:
    from wp4.constants import DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, POLLUTANTS
except ImportError:
    from constants import DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, POLLUTANTS


def _apply_fe_mask(ds):
    """
    Function to mask fire events from the dataset
    :param ds: xArray Dataset
    :param timestamp: timestamp of the fire event
    :param days:
    :return:
    """

    # initiate boolean layer named fire mask
    fire_mask = xr.open_dataset(Path(DATA_DIR_CAMS_AN).joinpath('fire_mask.nc')).copy()
    ds[list(ds.data_vars.keys())[0]] = ds[list(ds.data_vars.keys())[0]].where(cond=fire_mask['fire_mask'])

    return ds


def get_temporal_baseline(fe_lat, fe_long, timestamp, days, pollutant, years=[2015, 2016, 2017, 2018, 2019, 2020, 2021],
                          fire_mask=False):
    """
    :param fe_lat: float, Latitude of the fire event
    :param fe_long: float, Longitude of the fire event
    :param timestamp: Datetime timestamp
    :param days: number of days leading up and after the fire events for which to derive the baseline
    :param pollutant: pollutant for which to generate the baseline, options are:'CO','O3','NO','NO2','PM25','PM10','SO2'
    :param years: years to be included for the baseline calculation
    :param fire_mask: Boolean, if true, a mask is applied to the CAMS dataset used for the spatial baseline to remove
    CAMS pixels potentially impacted by these events
    :return:Pandas dataframe containing the baseline values for the period surrounding a fire event
    """

    # load dataset
    pollutant_variable_name = POLLUTANTS[pollutant]['CAMS']
    ds_cams = xr.open_dataset(f"{DATA_DIR_CAMS_AN}/{pollutant_variable_name}.nc").copy()

    if pd.to_datetime(timestamp.round('h')) not in ds_cams.time:
        print(f'No CAMS data available for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}.')
        return None

    year = timestamp.year  # Get the year that fire event took place

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

    # mask data for other fire event that occured during the period of time used for the FE analysis
    if fire_mask:
        ds_cams = _apply_fe_mask(ds_cams)

    # Deal with Leap Years
    if not ((timestamp.day == 29) and (timestamp.month == 2)):
        # If the fire event did not occur on a leap day, data for all leap days are removed from the CAMS dataset
        ds_bs = ds_cams.sel(time=~((ds_cams.time.dt.month == 2) & (ds_cams.time.dt.day == 29)))
        # Select data from the nearest CAMS cell to the fire event
        ds_bs = ds_bs.sel(latitude=fe_lat, longitude=fe_long, method='nearest')

        # Creates a list containing datasets for the specified timerange for the same dates that are being observed for
        # the fire event for all available years, excluding the year the fire event took place
        ds_other_years_subset = [ds_bs.sel(
            time=slice(
                timestamp.replace(minute=0, year=y) - timedelta(days=days),
                timestamp.replace(minute=0, year=y) + timedelta(days=days)
            )) for y in years if y != year]

    else:
        # Select data from the nearest CAMS cell to the fire event
        ds_bs = ds_cams.sel(latitude=fe_lat, longitude=fe_long, method='nearest')

        # Remove the year of the fire event from the list of years to use for the temporal baseline
        years = years.remove(year)

        # In case the fire event did take place on a leap day, the first of May is used as replacement during non leap years
        ds_other_years_subset = [ds_bs.sel(
            time=slice(
                timestamp.replace(minute=0, year=y) - timedelta(days=days),
                timestamp.replace(minute=0, year=y) + timedelta(days=days)
            )) if calendar.isleap(y) else ds_bs.sel(
            time=slice(
                timestamp.replace(minute=0, day=1, month=3, year=y) - timedelta(days=days),
                timestamp.replace(minute=0, day=1, month=3, year=y) + timedelta(days=days)
            )) for y in years]

    # remove any datasets that do not contain any data; apply squeeze to remove with all dimensions with a length of 1.
    ds_other_years = [d.squeeze() for d in ds_other_years_subset if (d.time.size != 0)]

    # Derive the temporal baseline for the period of interest by averaging the concentration levels for each timestamp
    mean_other_years = [
        pd.DataFrame(
            prev_ds[pollutant_variable_name].to_pandas(), columns=[f'{pollutant_variable_name}']
        ).reset_index()[f'{pollutant_variable_name}'] for prev_ds in ds_other_years]
    df_other_years_mean = list(pd.concat(mean_other_years, axis=1).mean(axis=1))
    df_other_years_median = list(pd.concat(mean_other_years, axis=1).median(axis=1))
    df_other_years_lower_quartile = list(pd.concat(mean_other_years, axis=1).quantile(0.25, axis=1))
    df_other_years_upper_quartile = list(pd.concat(mean_other_years, axis=1).quantile(0.75, axis=1))
    df_pixel_data_std = list(pd.concat(mean_other_years, axis=1).std(axis=1))

    def hour_from_event(x):
        """Function to calculate hour difference from a fire event"""

        x_time = x.values[0]  # select the timestamp

        if type(x_time) == datetime64:   # if the datetime is a numpy datetime64 object convert to datetime object
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

    # Combine the index, temporal baseline and fire event concentration levels into a single pandas dataframe
    df_baseline = pd.DataFrame({
        'time': df_index_time,
        'hour_from_event': df_hour_from_event,
        'temporal_baseline_mean': df_other_years_mean,
        'temporal_baseline_median':df_other_years_median,
        'temporal_baseline_lower_quartile':df_other_years_lower_quartile,
        'temporal_baseline_upper_quartile':df_other_years_upper_quartile,
        'temporal_baseline_std':df_pixel_data_std,
        'fire_event': list(ds_fe.squeeze()[pollutant_variable_name].to_pandas())
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

    pollutant = 'PM10'  # Can be any of 'CO', 'O3', 'NO', 'NO2', 'PM25', 'PM10', 'SO2'

    for ind, fe in df_fire_events.head(1).iterrows():

        try:
            df_baseline = get_temporal_baseline(
                fe_lat=fe['latitude'],
                fe_long=fe['longitude'],
                timestamp=fe['datetime'],
                days=5,
                pollutant=pollutant,
                fire_mask=True
            )

            print(df_baseline)

        except Exception as e:
            print(f'Skipping fire {ind} because of the following error: {e}')


if __name__ == '__main__':
    test()
