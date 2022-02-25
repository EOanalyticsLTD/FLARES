"""
Program to generate csv files containing deviation from specified baseline for all fire events pulled from database.
"""

# load the required packages
import os
import psycopg2
import pandas as pd
from pathlib import Path
import argparse
import time

# Local imports
try:
    from wp4.constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, DATA_DIR_PLOTS
    from wp4.baseline.spatial import get_spatial_baseline
    from wp4.baseline.temporal import get_temporal_baseline
    from wp4.baseline.spatiotemporal import get_spatiotemporal_baseline
except ImportError:
    from constants import POLLUTANTS, DB_HOST, DB_NAME, DB_USER, DB_PASS, DATA_DIR_CAMS_AN, DATA_DIR_PLOTS
    from baseline.spatial import get_spatial_baseline
    from baseline.temporal import get_temporal_baseline
    from baseline.spatiotemporal import get_spatiotemporal_baseline


def load_fire_events(query):
    # initiate connection to database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

    # Load as a dataframe
    df_fire_events = pd.read_sql_query(query, con=conn).rename(columns={'st_x': 'longitude', 'st_y': 'latitude'})

    # close the connection
    conn.close()

    if len(df_fire_events) == 0:
        print('No fire events found matching your parameters')
        return None

    return df_fire_events


def create_dataframe(main_df, df_to_join, index, new_col_name):
    """
    Function to join baseline deviation dataframes into a single dataframe
    :param main_df: The main pandas dataframe, to which all other deviation dataframes are joined
    :param df_to_join: The dataframe to join with the main dataframe
    :param index: the hour_from_event data from the baseline dataframe, this will serve as the join column
    :param new_col_name: name of the new column, storing the baseline deviation information
    :return:
    """

    if main_df is None:
        main_df = pd.DataFrame({
            'hours_from_fe': index,
            new_col_name: df_to_join}
        )
    else:
        df_to_join = pd.DataFrame({
            'hours_from_fe': index,
            new_col_name: df_to_join}
        )
        main_df = pd.merge(
            main_df, df_to_join,
            left_on='hours_from_fe',
            right_on='hours_from_fe'
        )

    return main_df


def run_analysis(df_fire_events, pollutants, baseline, days=5, source=None, ref=None, firetype=None):

    for pol in pollutants:

        # some variables to store the deviation dataframes in
        pollutant_difference = None
        pollutant_difference_percent = None

        print(f'Processing started for: {pol}')

        for ind, fe in df_fire_events.iterrows():  # iterate over the fire event dataframe

            tic = time.time()

            if baseline == 'temporal':
                # get the baseline information
                df_baseline = get_temporal_baseline(
                    fe_lat=fe['latitude'],
                    fe_long=fe['longitude'],
                    timestamp=fe['datetime'],
                    days=days,
                    pollutant=pol,
                )
                if df_baseline is None:
                    continue  # skip the fire event in case no baseline could be retrieved

                # Calculate difference between baseline and concentration levels during fe
                difference = df_baseline['fire_event'] - df_baseline['temporal_baseline_median']
                difference_percent = df_baseline[['temporal_baseline_median', 'fire_event']].pct_change(
                    axis='columns', periods=1)['fire_event'] * 100

            if baseline == 'spatial':
                df_baseline, _, _, _ = get_spatial_baseline(
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

                if df_baseline is None:
                    continue  # skip the fire event in case no baseline could be retrieved

                # Calculate difference between baseline and concentration levels during fe
                difference = df_baseline['fire_event'] - df_baseline['spatial_baseline_median']
                difference_percent = df_baseline[['spatial_baseline_median', 'fire_event']].pct_change(
                    axis='columns', periods=1)['fire_event'] * 100

            if baseline == 'spatiotemporal':
                df_baseline = get_spatiotemporal_baseline(
                    fe_lat=fe['latitude'],
                    fe_long=fe['longitude'],
                    timestamp=fe['datetime'],
                    pollutant=pol,
                    days=days,
                    min_distance_km=0,
                    max_distance_km=500,
                    mask_ocean=True
                )

                if df_baseline is None:
                    continue  # skip the fire event in case no baseline could be retrieved

                # Calculate difference between baseline and concentration levels during fe
                difference = df_baseline['fire_event'] - df_baseline['spatiotemporal_baseline_median']
                difference_percent = df_baseline[['spatiotemporal_baseline_median', 'fire_event']].pct_change(
                    axis='columns', periods=1)['fire_event'] * 100

            # create/combine dataframe containing the deviation for each fire event as a column
            pollutant_difference = create_dataframe(
                pollutant_difference,
                difference,
                list(df_baseline['hour_from_event']),
                ind,
            )

            pollutant_difference_percent = create_dataframe(
                pollutant_difference_percent,
                difference_percent,
                list(df_baseline['hour_from_event']),
                ind,
            )

            toc = time.time()
            time_difference = toc - tic

            print(f'Iteration: {ind} of {len(df_fire_events)} completed. Runtime: {round(time_difference, 2)} s')

        if len(pollutant_difference > 0) and pollutant_difference is not None:
            save_as_csv(pol, pollutant_difference, pollutant_difference_percent, baseline,
                        source, ref, firetype)


def save_as_csv(pollutant, pollutant_difference, pollutant_difference_percent, baseline,
                source=None, ref=None, firetype=None):

    df_perc = pollutant_difference_percent.transpose()
    df_conc = pollutant_difference.transpose()
    csv_loc = Path(DATA_DIR_PLOTS).joinpath(f'analysis/baseline_deviation/{baseline}/csv')

    if not os.path.exists(csv_loc):
        os.makedirs(csv_loc)

    filename = f'{pollutant}_'

    if source is not None:
        filename += f'{source}_'

    if ref is not None:
        for r in ref:
            filename += f'{r}_'

    if firetype is not None:
        for t in firetype:
            filename += f'{t}_'

    df_perc.to_csv(f'{csv_loc}/{filename}percentage.csv')
    df_conc.to_csv(f'{csv_loc}/{filename}concentration.csv')


def main(options):
    """
    Main function - processes the arguments
    """

    pollutants = options.pollutants
    baseline = options.baseline

    for pol in pollutants:
        if pol == 'all':
            pollutants = list(POLLUTANTS.keys())
        elif pol not in list(POLLUTANTS.keys()):
            print(
                f'Parameter for pollutants {pol} is unknown, please use one of {list(POLLUTANTS.keys())}')
            return None

    print(f'Pollutants: {pollutants}')
    print(f'Baseline: {baseline}')
    if baseline not in ['spatial', 'temporal', 'spatiotemporal']:
        print(
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

    print(f'Source: {options.source}')
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
        print(f'Reference: {options.ref}')
        for r in options.ref:
            if r not in list(references[options.source].keys()):
                print(
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
        print(f'Type: {options.type}')
        for t in options.type:
            if t not in list(fire_type.keys()):
                print(
                    f'Parameter for source "{t}" is unknown, please use one of {list(fire_type.keys())}')
                return None



    df_fire_events = load_fire_events(query)

    if df_fire_events is None:
        return None

    if options.days:
        run_analysis(df_fire_events, pollutants, baseline, days=options.days,
                     source=options.source, ref=options.ref, firetype=options.type)
    else:
        run_analysis(df_fire_events, pollutants, baseline, days=5,
                     source=options.source, ref=options.ref, firetype=options.type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command line tool to run the baseline deviation analysis')

    parser.add_argument(
        "-p", '--pollutants', type=str, nargs='+',
        help='A list of pollutant keys: "PM25","PM10","SO2","CO", "O3", "NO", "NO2"', required=True)
    parser.add_argument(
        "-b", '--baseline', type=str, help='The baseline to use: "spatial", "temporal", "spatiotemporal"',
        required=True
    )
    parser.add_argument("-s", "--source", type=str, help='The source of the fire event: "fb", "sat", "media"',
                        required=True)
    parser.add_argument("-r", "--ref", type=str, nargs='+', help="The specific reference of the source")

    parser.add_argument("-t", "--type", nargs='+', help="Define fire types, if selecting data from fire brigade")
    parser.add_argument("-d", "--days", help="Set the number of days to use for the analysis")

    args = parser.parse_args()
    main(args)
