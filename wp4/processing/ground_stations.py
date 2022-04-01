# General
import datetime as dt
import pandas as pd
import re
import datetime
import json

# Scraping
import bs4 as bs
import requests

# Database
import psycopg2

# Local imports
try:
    from wp4.constants import DB_HOST, DB_NAME, DB_USER, DB_PASS
except ImportError:
    from constants import DB_HOST, DB_NAME, DB_USER, DB_PASS


def get_closest_active_epa_ground_station(lat, long, pollutant, quantity=3):
    """
        Function that extracts closest active EPA ground station
        :param lat: float, Latitude
        :param long: float, Longitude
        :param pollutant: string, string indicating the pollutant for which data has to be extracted. Available pollutants:
        co_conc, o3_conc, no2_conc, pm2p5_conc, pm10_conc, so2_conc, no_conc.
        :return: dataframe containing details about the closest ground station.
    """

    pollutant_columns = { # dict with the pollutants names as used for historical data in the database
        'co_conc': 'co',
        'o3_conc': 'o3',
        'no2_conc': 'no2',
        'pm2p5_conc': 'pm25',
        'pm10_conc': 'pm10',
        'so2_conc': 'so2',
        'no_conc': 'no',
    }

    if pollutant == 'no_conc':
        return None

    pollutant_col = pollutant_columns[pollutant]

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)  # connect to the database

    df = pd.read_sql_query(  # Select the available information for the closest ground station
        f"""SELECT  *,
                    ST_X(geometry),
                    ST_Y(geometry),
                    ST_Distance(ST_SetSRID(ST_Point({long}, {lat}), 4326), ST_SetSRID(geometry, 4326), true) AS distance
                    FROM ground_stations
                    WHERE ground_stations.{pollutant_col} = true
                    ORDER BY ST_SetSRID(geometry, 4326)  <-> ST_SetSRID(ST_Point({long}, {lat}), 4326) 
                    LIMIT {quantity};""",
        conn)
    conn.close()

    df = df.rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})

    return df


def get_epa_data(epa_code, timestamp, pollutant, days=7):
    """
            Function that scrapes pollutant concentration data from AirQuality.ie for a given ground station and
            time period
            :param df: dataframe containing the details of the ground station for which the data is to be scraped
            :param timestamp: datetime timestamp
            :param pollutant: string, string indicating the pollutant for which data has to be extracted.
            Available pollutants: co_conc, o3_conc, no2_conc, pm2p5_conc, pm10_conc, so2_conc, no_conc.
            :return: dataframe containing pollution data scraped from AirQuality.ie
        """

    pollutants = {  # dict with the pollutants names as used for historical data in the database
        'co_conc': 'CO',
        'o3_conc': 'O3',
        'no2_conc': 'NO2',
        'pm2p5_conc': 'PM2.5',
        'pm10_conc': 'PM10',
        'so2_conc': 'SO2',
    }

    # Set the date range for the period during which the fire event took place
    begin_fire_event = timestamp - dt.timedelta(days=days)
    end_fire_event = timestamp + dt.timedelta(days=days)

    start_date_string = begin_fire_event.strftime('%d+%b+%Y')  # Convert to string to place in the url
    end_date_string = end_fire_event.strftime('%d+%b+%Y')

    # construct url to scrape
    url = f"https://airquality.ie/readings?station={epa_code}&dateFrom={start_date_string}&dateTo={end_date_string}"

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept-Encoding': 'identity',
    }

    page = requests.get(url, headers=headers)  # request the page

    soup = bs.BeautifulSoup(page.text, 'lxml') # parse the response

    graph = soup.findAll(
        'script',
        attrs={
            'type': 'text/javascript'}
    )  # find all the elements of text/javascript type

    if len(graph) == 0:
        return None  # no elements were found, so no data available

    # Currently the data shows in the graph is found in the last element text/javascript
    test = bs.BeautifulSoup(graph[-1].contents[0], 'lxml')

    data = {}

    # Use regex to select the pollutant data from the javascript parsed from the page
    names = re.findall('name:\"([A-Za-z0-9\.]*)\"', test.text.replace(
        "\n", "").replace(
        "\'", '"').replace(
        " ", "").replace("null", "None"), re.DOTALL)

    data_columns = re.findall("data:(\[[^/]*)//endforeach]", test.text.replace(
        "\n", "").replace(
        "\'", '"').replace(
        " ", "").replace("null", "None"), re.DOTALL)

    for ind, name in enumerate(names): # remove some unwanted JS and convert string to python list
        data[name] = eval(data_columns[ind].replace('Date.UTC', '') + ']')

    for key in data:  #  construct a dictionary with the data
        data[key] = {datetime.datetime(year=x[0][0], month=(x[0][1]) + 1, day=x[0][2], hour=x[0][3]): x[1] for x in
                     data[key]}

    try:
        # convert to pandas dataframe, convert datatype to float and return the df
        pol_name = pollutants[pollutant]
        data = pd.DataFrame(index=list(data[pol_name].keys()), data=list(data[pol_name].values()), columns=[pol_name])
        data[pol_name] = data[pol_name].astype(float)
        if pol_name == 'CO':
            data[pol_name] = data[pol_name] * 1000  # CO values need to be converted, are given in mg, no microgram

        return data
    except:
        return None


def get_closest_ground_station_historical_data(lat, long, timestamp, pollutant, quantity=3, days=7):
    """
    Function that extracts pollutant measurements from the database for the nearest ground station based on location and
    time and desired pollutant
    :param lat: float, Latitude
    :param long: float, Longitude
    :param timestamp: datetime timestamp
    :param pollutant: string, string indicating the pollutant for which data has to be extracted. Available pollutants:
    co_conc, o3_conc, no2_conc, pm2p5_conc, pm10_conc, so2_conc, no_conc.
    :param quantity: Number of ground stations to select
    :return: df_data: Dataframe containing the measurements taken from the closest available ground stations, distance:
    distance to the closest ground station, name: Official EPA name of the ground station, if available.
    """

    pollutant_columns = { # dict with the pollutants names as used for historical data in the database
        'co_conc': 'co',
        'o3_conc': 'o3',
        'no2_conc': 'no2',
        'pm2p5_conc': 'pm25',
        'pm10_conc': 'pm10',
        'so2_conc': 'so2',
        'no_conc': 'no',
    }

    ppb_to_micro = {  # some data needs to be converted from ppb, this dict contains the conversion factors
        'o3_conc': 2,
        'no2_conc': 1.88,
        'no_conc': 1.25,
    }

    pollutant_col = pollutant_columns[pollutant]  # store the database name of the pollutant in separate variable

    # Connect to the database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    df = pd.read_sql_query(  # selects the three closests ground stations to the location of the fire event
        f"""SELECT  *,
                    ST_Distance(ST_SetSRID(ST_Point({long}, {lat}), 4326), ST_SetSRID(geometry, 4326), true) AS distance
                    FROM ground_stations
                    WHERE (ground_stations.{pollutant_col}_hist = true)
                    ORDER BY ST_SetSRID(geometry, 4326)  <-> ST_SetSRID(ST_Point({long}, {lat}), 4326) 
                    LIMIT {quantity};""",
        conn)

    conn.close()

    # Set the date range for the period during which the fire event took place
    begin_fire_event = timestamp - dt.timedelta(days=days)
    end_fire_event = timestamp + dt.timedelta(days=days)

    start_date_string = begin_fire_event.strftime('%Y-%m-%d %H')  # Convert to string to place in SQL query
    end_date_string = end_fire_event.strftime('%Y-%m-%d %H')

    results = {}

    for ind, row in df.iterrows(): # go over each of the closest ground stations

        # make some adjustments to fit ground station column names used in the database
        col_name = row['data_name'].lower().replace(' ', '_').replace("'", '')
        distance = row['distance']  # distance from the fire event in meters

        if row['epa_name'] is not None:
            name = row['epa_name']  # use the officual EPA name if available (taken from AirQuality.ie)
        else:
            name = row['data_name'].replace('_', ' ').title()  # if no official EPA name is available use the data name

        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        df_data = pd.read_sql_query(  # Select the data for the station for the fire event period
            f"""SELECT datetime, {col_name} FROM ground_meas_{pollutant.replace('_conc', '')}
                    WHERE datetime BETWEEN '{start_date_string}:00:00'::timestamp
                 AND '{end_date_string}:00:00'::timestamp;""",
            conn)
        conn.close()

        if df_data.empty: # if no data was retrieved try the next ground station
            continue

        if df_data[col_name].isnull().sum() / len(df_data[col_name]) <= .5:  # at least 50% of the values are not null
            df_data = df_data.rename(columns={col_name: 'ground_station_data'})

            if pollutant in ['o3_conc', 'no2_conc', 'no_conc']:  # apply conversion if needed
                df_data['ground_station_data'] = df_data['ground_station_data'] * ppb_to_micro[pollutant]

            df_data = df_data.rename(columns={'datetime':'time'})

            results[ind] = {'data':df_data, 'distance':distance, 'name': name}
        else:
            continue

    if len(results) > 0:
        return results
    else:
        return None  # if no data was retrieved return None


def test():
    get_epa_data("EPA-44", datetime.datetime(year=2021, month=4, day=15, hour=15), 'pm2p5_conc')


if __name__ == '__main__':
    test()