import os, glob
import psycopg2
import pandas as pd
import xarray as xr
import dask
from pathlib import Path
from datetime import datetime, timedelta
from numpy import datetime64

# Local imports
try:
    from wp4.constants import POLLUTANTS, DATA_DIR_CAMS_AN, DATA_DIR_CAMS_RE, EXTENTS, DB_HOST, DB_NAME, DB_USER, DB_PASS
except ImportError:
    from constants import POLLUTANTS, DATA_DIR_CAMS_AN, DATA_DIR_CAMS_RE, EXTENTS,  DB_HOST, DB_NAME, DB_USER, DB_PASS

# Dask chunk config, to prevent warnings. Setting this to True seems to make processing a small bit faster.
dask.config.set({"array.slicing.split_large_chunks": True})


def create_dataset(pollutant: str, pref: str = "reanalysis", years: list = None):
    """
    @param pollutant: pollutant to create dataset for - Options are: 'CO','O3','NO','NO'2,'PM'25,'PM'10,'SO2'.
    @param pref: String specifying which dataset has preference when creating the combined dataset. You can choose the
    Reanalysis dataset -> "reanalysis" or the NRT analysis dataset -> "analysis"
    @param years: list of years to select for the dataset, defaults to all years between 2015 to 2021
    @return: returns the best dataset based on the years given and the preferred dataset
    """

    if pref not in ['reanalysis', 'analysis']:
        raise ValueError(f'Unknown dataset: {pref}. You can choose "reanalysis" or "analysis')

    if years is None:
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

    datasets_an = os.listdir(Path(DATA_DIR_CAMS_AN).joinpath(pollutant))
    datasets_re = os.listdir(Path(DATA_DIR_CAMS_RE).joinpath(pollutant))

    years_available_an = [int(x[-7:-3]) for x in datasets_an]
    years_available_re = [int(x[-7:-3]) for x in datasets_re]

    filenames = []

    if pref == 'reanalysis':
        missing_years = [year for year in years if (year not in years_available_re) and year in years_available_an]
        available_years = [year for year in years if (year not in missing_years) and year in years_available_re]

        for year in available_years:
            filenames += [f"{DATA_DIR_CAMS_RE}/{pollutant}/cams_reanalysis_{pollutant}_{year}.nc"]

        for missing_year in missing_years:
            filenames += [f"{DATA_DIR_CAMS_AN}/{pollutant}/cams_nrt_analysis_{pollutant}_{missing_year}.nc"]
    else:
        missing_years = [year for year in years if (year not in years_available_an) and year in years_available_re]
        available_years = [year for year in years if (year not in missing_years) and year in years_available_an]

        for year in available_years:
            filenames += [f"{DATA_DIR_CAMS_AN}/{pollutant}/cams_nrt_analysis_{pollutant}_{year}.nc"]

        for missing_year in missing_years:
            filenames += [f"{DATA_DIR_CAMS_RE}/{pollutant}/cams_reanalysis_{pollutant}_{missing_year}.nc"]

    if len(filenames):
        final_dataset = xr.open_mfdataset(filenames)
    else:
        # print(f'No CAMS data available for {pollutant}')
        return None

    if 'level' in list(final_dataset.dims):
        final_dataset = final_dataset.squeeze(drop=True)

    return final_dataset


def get_timeseries_fire(pollutant, timestamp, fe_long, fe_lat, days=5, years=None, extent='IRELAND'):

    if not EXTENTS[extent]['WEST'] <= fe_long <= EXTENTS[extent]['EAST']:
        # print(
        #     f'Longitude: {fe_long} is outside of the study area extent: {EXTENTS[extent]["WEST"]} - {EXTENTS[extent]["EAST"]}')
        return None

    if not EXTENTS[extent]['SOUTH'] <= fe_lat <= EXTENTS[extent]['NORTH']:
        # print(
        #     f'Latitude: {fe_lat} is outside of the study area extent: {EXTENTS[extent]["SOUTH"]} - {EXTENTS[extent]["NORTH"]}')
        return None

    # check years param
    if years is None:
        years=[2015, 2016, 2017, 2018, 2019, 2020, 2021]

    # For each year that the fire did not take place
    # load dataset
    pollutant_variable_name = POLLUTANTS[pollutant]['CAMS']
    ds_cams = create_dataset(pollutant, years=years).copy()[pollutant_variable_name]

    if pd.to_datetime(timestamp.round('h')) not in ds_cams.time:
        # print (f'No CAMS data available for {POLLUTANTS[pollutant]["FULL_NAME"]} for timestamp: {timestamp}.')
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

    ar_fire_event = ds_fe.squeeze(drop=True).to_pandas()

    if df_hour_from_event.index.size != len(ar_fire_event):

        # combined dataframe based on datetime index, missing dates will have NA col values
        df_fill = pd.DataFrame(index=df_index_time)
        df_fe = pd.DataFrame(ds_fe.squeeze().to_pandas(), columns=[ds_fe.name], index=ds_fe.time)
        df_filled = df_fill.join(df_fe)

        # convert to list
        ar_fire_event = df_filled[ds_fe.name].tolist()

    df = pd.DataFrame({
        'time': df_index_time,
        'hour_from_event': df_hour_from_event.tolist(),
        'fire_event': ar_fire_event
    })

    return df


if __name__ == '__main__':

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    query = """
                SELECT id, datetime, ST_X(geometry), ST_Y(geometry), source, location, reference, type, info
                FROM public.all_fire_events
            """

    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})

    conn.close()

    pollutant = 'NO2'  # Can be any of 'CO', 'O3', 'NO', 'NO2', 'PM25', 'PM10', 'SO2'

    for ind, fe in df_fire_events.iterrows():

        try:
            df_baseline = get_timeseries_fire(
                fe_lat=fe['latitude'],
                fe_long=fe['longitude'],
                timestamp=fe['datetime'],
                days=5,
                pollutant=pollutant,
            )

            print(df_baseline)

        except Exception as e:
            raise
            print(f'Skipping fire {ind} because of the following error: {e}')

