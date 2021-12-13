"""
Functions for the visualization of wind data
"""

import pandas as pd
import plotly.express as px


def create_windrose(df, title=None):
    """
    Function to create windrose from wind speed and direction data.
    Most of the code was taken from the following community post:

    https://community.plotly.com/t/wind-rose-with-wind-speed-m-s-and-direction-deg-data-columns-need-help/33274/3
    """

    # creating bins for magnitudes and directions

    bins_mag = [0, 0.5, 1.5, 3.0, 5.0, 8, 11, 14, 20, 100]
    bins_mag_labels = ['0-0.5 m/s', '0.5-1.5 m/s',
                       '1.5-3 m/s', '3-5 m/s', '5-8 m/s',
                       '8-11 m/s', '11-14 m/s', '14-20 m/s', '20+ m/s']

    bins_dir = [0, 11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75, 191.25, 213.75, 236.25, 258.75, 281.25,
                303.75, 326.25, 348.75, 360.00]
    bins_dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW',
                       'NNW', 'North']

    # bin magnitude and direction from the meteo dataframe
    df['mag_binned'] = pd.cut(df['wind_speed'], bins_mag, labels=bins_mag_labels)
    df['dir_binned'] = pd.cut(df['wind_direction'], bins_dir, labels=bins_dir_labels)

    dfe = df[['mag_binned', 'dir_binned', 'u10']].copy()
    dfe.rename(columns={'u10': 'freq'}, inplace=True)  # changing the last column to represent frequencies
    g = dfe.groupby(['mag_binned', 'dir_binned']).count()  # grouping
    g.reset_index(inplace=True)
    g['percentage'] = g['freq'] / g['freq'].sum()
    g['percentage%'] = g['percentage'] * 100
    g['percentage%'] = g['percentage%'].round(2)
    g['Magnitude [m/s]'] = g['mag_binned']
    g = g.replace(r'North', 'N', regex=True)  # replacing remaining Norths with N

    # Create the plotly figure from with the data

    fig = px.bar_polar(
        g,
        r="percentage%",
        theta="dir_binned",
        color="Magnitude [m/s]",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )

    fig.update_layout(polar_radialaxis_ticksuffix='%', title=title)

    return fig