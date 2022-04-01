#!/usr/bin/env python3
"""
Quick program to generate csv files containing pollutant concentration/baseline information for all fires in the ground
truth database.
"""

# load the required packages
import os
import psycopg2
import pandas as pd
from pathlib import Path
import argparse
import datetime
import logging

# Local imports
try:
    from wp4.constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, DATA_DIR_PLOTS
    from wp4.processing.helpers import get_timeseries_fire
    from wp4.baseline.spatial import get_spatial_baseline
    from wp4.baseline.temporal import get_temporal_baseline
    from wp4.baseline.spatiotemporal import get_spatiotemporal_baseline
except ImportError:
    from constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, DATA_DIR_PLOTS
    from processing.helpers import get_timeseries_fire
    from baseline.spatial import get_spatial_baseline
    from baseline.temporal import get_temporal_baseline
    from baseline.spatiotemporal import get_spatiotemporal_baseline

OUTPUT_DIR = Path(DATA_DIR_PLOTS).joinpath('csv_database')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_fire_events(query):
    """Load fire events as pandas dataframe from the database"""
    # initiate connection to database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    # Load as a dataframe
    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})

    # close the connection
    conn.close()

    # in case of no matches
    if len(df_fire_events) == 0:
        print('No fire events found matching your parameters')
        return None

    return df_fire_events


def run_analysis(df_fire_events, pollutants, baseline, days=5):
    """Runs the analysis for all the fire events in given dataframe"""

    # loop through the pollutants
    for pol in pollutants:
        logging.info(f'Processing started for: {pol}')
        start_datetime = datetime.datetime.now()
        logging.info(f"Processing started on: {start_datetime.strftime('%b %d -  %H:%M')}")

        # variable to count the completed iterations
        completed = 0

        for ind, fe in df_fire_events.iterrows():  # iterate over the fire event dataframe

            print(fe['datetime'])

            # initiate the dataframes that will contain the data
            df_fire_event_conc = None
            df_fire_temporal_baseline = None
            df_fire_spatial_baseline = None
            df_fire_spatiotemporal_baseline = None

            # for spatial baseline, this will also create the fire event concentration dataframe
            if baseline == "fire_only":

                df_fe = get_timeseries_fire(
                    fe_lat=fe['latitude'],
                    fe_long=fe['longitude'],
                    timestamp=fe['datetime'],
                    days=days,
                    pollutant=pol,
                )

                if df_fe is None:
                    continue


                if df_fire_event_conc is None:

                    metadata = fe.index.tolist()
                    cols = metadata + df_fe['hour_from_event'].tolist()
                    df_fire_event_conc = pd.DataFrame(columns=cols)

                df_fire_event_conc.loc[len(df_fire_event_conc)] = fe.tolist() + df_fe['fire_event'].round(6).tolist()

            elif baseline == 'spatial':

                try:
                    df_baseline_spatial, _, _, _ = get_spatial_baseline(
                        fe_lat=fe['latitude'],
                        fe_long=fe['longitude'],
                        timestamp=fe['datetime'],
                        days=days,
                        pollutant=pol,
                        meteo_dataset='MERA',
                        min_distance_km=50,
                        max_distance_km=200,
                        number_of_neighbours=50,
                        mask_ocean=True,
                    )
                except Exception as e:
                    logging.debug(f'Iteration {ind} skipped because of  {e}')
                    continue

                if df_baseline_spatial is None:
                    continue

                if df_fire_spatial_baseline is None:

                    metadata = fe.index.tolist()
                    cols = metadata + ['baseline_stat'] + df_baseline_spatial['hour_from_event'].tolist()
                    df_fire_spatial_baseline = pd.DataFrame(columns=cols)

                df_fire_spatial_baseline.loc[len(df_fire_spatial_baseline)] = fe.tolist() + ['median'] + df_baseline_spatial['spatial_baseline_median'].round(6).tolist()
                df_fire_spatial_baseline.loc[len(df_fire_spatial_baseline)] = fe.tolist() + ['mean'] + df_baseline_spatial['spatial_baseline_mean'].round(6).tolist()
                df_fire_spatial_baseline.loc[len(df_fire_spatial_baseline)] = fe.tolist() + ['25p'] + df_baseline_spatial['spatial_baseline_lower_quartile'].round(6).tolist()
                df_fire_spatial_baseline.loc[len(df_fire_spatial_baseline)] = fe.tolist() + ['75p'] + df_baseline_spatial['spatial_baseline_upper_quartile'].round(6).tolist()
                df_fire_spatial_baseline.loc[len(df_fire_spatial_baseline)] = fe.tolist() + ['stddev'] + df_baseline_spatial['spatial_baseline_std'].round(6).tolist()

            elif baseline == 'temporal':

                try:
                    df_baseline_temporal = get_temporal_baseline(
                        fe_lat=fe['latitude'],
                        fe_long=fe['longitude'],
                        timestamp=fe['datetime'],
                        days=days,
                        pollutant=pol,
                        fire_mask=False,
                    )
                except Exception as e:
                    logging.debug(f'Iteration {ind} skipped because of  {e}')
                    continue

                if df_baseline_temporal is None:
                    continue

                if df_fire_temporal_baseline is None:

                    metadata = fe.index.tolist()
                    cols = metadata + ['baseline_stat'] + df_baseline_temporal['hour_from_event'].tolist()
                    df_fire_temporal_baseline = pd.DataFrame(columns=cols)

                df_fire_temporal_baseline.loc[len(df_fire_temporal_baseline)] = fe.tolist() + ['median'] + df_baseline_temporal['temporal_baseline_median'].round(6).tolist()
                df_fire_temporal_baseline.loc[len(df_fire_temporal_baseline)] = fe.tolist() + ['mean'] + df_baseline_temporal['temporal_baseline_mean'].round(6).tolist()
                df_fire_temporal_baseline.loc[len(df_fire_temporal_baseline)] = fe.tolist() + ['25p'] + df_baseline_temporal['temporal_baseline_lower_quartile'].round(6).tolist()
                df_fire_temporal_baseline.loc[len(df_fire_temporal_baseline)] = fe.tolist() + ['75p'] + df_baseline_temporal['temporal_baseline_upper_quartile'].round(6).tolist()
                df_fire_temporal_baseline.loc[len(df_fire_temporal_baseline)] = fe.tolist() + ['stddev'] + df_baseline_temporal['temporal_baseline_std'].round(6).tolist()

            elif baseline == 'spatiotemporal':
                try:
                    df_baseline_spatio_temporal = get_spatiotemporal_baseline(
                        fe_lat=fe['latitude'],
                        fe_long=fe['longitude'],
                        timestamp=fe['datetime'],
                        pollutant=pol,
                        days=days,
                        min_distance_km=0,
                        max_distance_km=500,
                        mask_ocean=True
                    )
                except Exception as e:
                    logging.debug(f'Iteration {ind} skipped because of  {e}')
                    continue

                if df_baseline_spatio_temporal is None:
                    continue

                if df_fire_spatiotemporal_baseline is None:

                    metadata = fe.index.tolist()
                    cols = metadata + ['baseline_stat'] + df_baseline_spatio_temporal['hour_from_event'].tolist()
                    df_fire_spatiotemporal_baseline = pd.DataFrame(columns=cols)

                df_fire_spatiotemporal_baseline.loc[len(df_fire_spatiotemporal_baseline)] = fe.tolist() + ['median'] + df_baseline_spatio_temporal['spatiotemporal_baseline_median'].round(6).tolist()
                df_fire_spatiotemporal_baseline.loc[len(df_fire_spatiotemporal_baseline)] = fe.tolist() + ['mean'] + df_baseline_spatio_temporal['spatiotemporal_baseline_mean'].round(6).tolist()
                df_fire_spatiotemporal_baseline.loc[len(df_fire_spatiotemporal_baseline)] = fe.tolist() + ['25p'] + df_baseline_spatio_temporal['spatiotemporal_baseline_lower_quartile'].round(6).tolist()
                df_fire_spatiotemporal_baseline.loc[len(df_fire_spatiotemporal_baseline)] = fe.tolist() + ['75p'] + df_baseline_spatio_temporal['spatiotemporal_baseline_upper_quartile'].round(6).tolist()
                df_fire_spatiotemporal_baseline.loc[len(df_fire_spatiotemporal_baseline)] = fe.tolist() + ['stddev'] + df_baseline_spatio_temporal['spatiotemporal_baseline_std'].round(6).tolist()

            completed += 1

            if df_fire_event_conc is not None:

                if len(df_fire_event_conc) < 0:
                    continue

                df_fire_event_conc.to_csv(
                    OUTPUT_DIR.joinpath(f'{pol}_CAMS_fe_concentration.csv'),
                    index=False,
                    mode='a',
                    header=False,
                )

            if df_fire_temporal_baseline is not None:

                if len(df_fire_temporal_baseline) < 0:
                    continue

                df_fire_temporal_baseline.to_csv(
                    OUTPUT_DIR.joinpath(f'{pol}_CAMS_fe_temporal_baseline.csv'),
                    index=False,
                    mode='a',
                    header=True,
                )

            if df_fire_spatial_baseline is not None:

                if len(df_fire_spatial_baseline) < 0:
                    continue

                df_fire_spatial_baseline.to_csv(
                    OUTPUT_DIR.joinpath(f'{pol}_CAMS_fe_spatial_baseline.csv'),
                    index=False,
                    mode='a',
                    header=False,
                )

            if df_fire_spatiotemporal_baseline is not None:

                if len(df_fire_spatiotemporal_baseline) < 0:
                    continue

                df_fire_spatiotemporal_baseline.to_csv(
                    OUTPUT_DIR.joinpath(f'{pol}_CAMS_fe_spatiotemporal_baseline.csv'),
                    index=False
                )

            logging.info(f'Saving {baseline} data, currently {completed} fire events completed.')
        end_datetime = datetime.datetime.now()
        time_dif = end_datetime - start_datetime
        logging.info(f"Processing Completed on: {end_datetime.strftime('%b %d -  %H:%M')}")
        logging.info(f"Processing Time: {time_dif.seconds/60} minutes")


def main(options):
    """
    Main function - processes the arguments
    """

    pollutants = options.pollutants
    baseline = options.baseline

    # start logging.
    logging.basicConfig(
        filename=OUTPUT_DIR.joinpath(f'{baseline}.log'),
        filemode='a',
        format='%(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    for pol in pollutants:
        if pol == 'all':  # if option 'all' is given use all the pollutant keys
            pollutants = list(POLLUTANTS.keys())
        elif pol not in list(POLLUTANTS.keys()):
            logging.error(
                f'Parameter for pollutants {pol} is unknown, please use one of {list(POLLUTANTS.keys())}')
            return None

    logging.info(f'Pollutants: {pollutants}')
    logging.info(f'Baseline: {baseline}')

    # check the baseline string
    if baseline not in ['fire_only', 'spatial', 'temporal', 'spatiotemporal']:
        logging.error(
            f'Parameter for baseline {baseline} is unknown, please use one of "spatial", "temporal", "spatiotemporal"')
        return None

    query = """  
            SELECT id, datetime, ST_X(geometry), ST_Y(geometry), source, location, reference, type, info
            FROM public.all_fire_events
        """

    sources = {
        'fb': "WHERE source = 'Fire Brigade'",
        'sat': "WHERE source = 'Satellite'",
        'media': "WHERE source = 'Media'",
    }

    logging.info(f'Source: {options.source}')
    if options.source not in list(sources.keys()):
        print(
            f'Parameter for source {options.source} is unknown, please use one of {list(sources.keys())}')
        return None

    references = {
        'fb': {
            'blackstairs': "reference ='Blackstairs Mountains Fire Brigade'",
            'clare': "reference ='Clare Fire Brigade'",
            'leinster': "reference ='Leinster Fire Brigade'",
            'meath': "reference ='Meath Fire Brigade'",
            'leitrim': "reference ='Leitrim Fire Brigade'",
            'mayo': "reference ='Mayo Fire Brigade'",
            'offaly': "reference ='Offaly Fire Brigade'",
            'roscommon': "reference ='Roscommon Fire Brigade'",
            'sligo': "reference ='Sligo Fire Brigade'",
            'west': "reference ='West Cork Fire Brigade'",
            'westmeath': "reference ='Westmeath Fire Brigade'",
            'wicklow': "reference =Wicklow Fire Brigade'",
        },
        'sat': {
            'modis': "reference = 'Aqua' OR reference = 'Terra'",
            'aqua': "reference = 'Aqua'",
            'terra': "reference = 'Terra'",
            'suomi': "reference = 'Suomi NPP'",
            'noaa': "reference = 'NOAA-20'",
        },
    }

    if options.ref:
        logging.info(f'Reference: {options.ref}')
        for r in options.ref:
            if r not in list(references[options.source].keys()):
                logging.error(
                    f'Parameter for reference {r} is unknown, please use one of {list(references[options.source].keys())}')
                return None

    fire_type = {
        'bog': "type = 'Bog Fire'",
        'bush': "type = 'Bush Fire'",
        'grass': "type = 'Grass Fire'",
        'forest': "type = 'Forest Fire'",
        'gorse': "type = 'Gorse Fire'",
        'controlled': "type = 'Controlled Fire'",
    }

    if options.type:
        logging.info(f'Type: {options.type}')
        for t in options.type:
            if t not in list(fire_type.keys()):
                logging.error(
                    f'Parameter for source "{t}" is unknown, please use one of {list(fire_type.keys())}')
                return None

    if options.ref is None or options.source == 'media':
        query += sources[options.source]

    else:
        references = references[options.source]

        for ind, ref in enumerate(options.ref):

            source_string = references[ref.lower()]

            if ind == 0:
                query += f"""WHERE {source_string}"""
            else:
                query += f"""OR {source_string}"""

    if options.source == 'fb' and options.type:

        for ind, tp in enumerate(options.type):

            source_string = fire_type[tp.lower()]

            if ind == 0:
                query += f"""AND {source_string}"""
            else:
                query += f"""OR {source_string}"""

    logging.info(f'Query: {query}')

    df_fire_events = load_fire_events(query)

    if df_fire_events is None:
        logging.error('No fire events found in database')
        return None

    # try:
    run_analysis(df_fire_events, pollutants, baseline)
    # except Exception as e:
    #     logging.error(f'ERROR - Processing {baseline} aborted. ERROR: {e}')

    logging.info(f'Processing {baseline} completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command line tool to run the baseline deviation analysis')

    parser.add_argument(
        "-p", '--pollutants', type=str, nargs='+',
        help='A list of pollutant keys: "PM25","PM10","SO2","CO", "O3", "NO", "NO2"', required=True)
    parser.add_argument(
        "-b", '--baseline', type=str, help='The baseline to use: "fire_only","spatial", "temporal", "spatiotemporal"',
        required=True
    )
    parser.add_argument("-s", "--source", type=str, help='The source of the fire event: "fb", "sat", "media"',
                        required=True)
    parser.add_argument("-r", "--ref", type=str, nargs='+', help="The specific reference of the source")

    parser.add_argument("-t", "--type", nargs='+', help="Define fire types, if selecting data from fire brigade")

    parser.add_argument("-d", "--days", help="Set the number of days to use for the analysis")

    args = parser.parse_args()
    main(args)