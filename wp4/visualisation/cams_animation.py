"""
Function for the creation of GIF animation of an xArray dataset
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import glob
from PIL import Image

from flares_package.constants import DATA_DIR_PLOTS

OUTPUT_FOLDER = f'{DATA_DIR_PLOTS}/animation/'


def create_animation(ds, title, folder, colormap='RdYlGn_r', min_val=None, max_val=None,
                     font_size=18, duration=600, remove_plots=True):
    """"""

    if min_val is None:
        min_val = ds.min().data

    if max_val is None:
        max_val = ds.max().data

    for i in ds.time.to_pandas().items():
        i = i[0]
        hour = ds.sel(time=i)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.Orthographic(8.7, 49.9))
        ax.coastlines()

        hour.plot(
            transform=ccrs.PlateCarree(),
            robust=True,
            vmin=min_val,
            vmax=max_val,
            cmap=plt.get_cmap(colormap),
            extend='neither',
            facecolor="gray"
        )

        title_timestamp = i.strftime("%H:%M %d %b %Y")
        ax.set_title(f'{title} {title_timestamp}',
                     {'fontsize': font_size})

        if not os.path.exists(f'{OUTPUT_FOLDER}{folder}/'):
            os.makedirs(f'{OUTPUT_FOLDER}{folder}/', exist_ok=True)

        plt.savefig(f'{OUTPUT_FOLDER}{folder}/{i.strftime("%Y_%b_%d_%H_%M_")}.jpg')
        plt.close()

    fp_in = f'{OUTPUT_FOLDER}{folder}/*.jpg'

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

    img.save(
        fp=f'{OUTPUT_FOLDER}{folder}/animation_{title}.gif',
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=duration,
        loop=0,
    )

    if remove_plots:
        for f in glob.glob(fp_in):
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    return True