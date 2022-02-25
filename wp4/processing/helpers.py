import os, glob
import xarray as xr
import dask
from pathlib import Path

# Local imports
try:
    from wp4.constants import POLLUTANTS, DATA_DIR_CAMS_AN, DATA_DIR_CAMS_RE
except ImportError:
    from constants import POLLUTANTS, DATA_DIR_CAMS_AN, DATA_DIR_CAMS_RE

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
        print(f'No CAMS data available for {pollutant}')
        return None

    if 'level' in list(final_dataset.dims):
        final_dataset = final_dataset.squeeze(drop=True)

    return final_dataset
