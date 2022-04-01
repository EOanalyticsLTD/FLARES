# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:38:32 2022

@author: guyse
"""

import os, sys, subprocess, datetime
try: # This is included as the module may not properly install in Anaconda.
    import ieo
except:
    # ieodir = os.getenv('IEO_INSTALLDIR')
    # if not ieodir:
    if os.path.isfile('../ieo/ieo.py'):
        ieodir = '../ieo'
    else:
        print('Error: IEO failed to load. Please input the location of the directory containing the IEO installation files.')
        ieodir = input('IEO installation path: ')
    if os.path.isfile(os.path.join(ieodir, 'ieo.py')):
        sys.path.append(ieodir)
        import ieo
        import S3ObjectStorage
    else:
        print('Error: that is not a valid path for the IEO module. Exiting.')
        sys.exit()

workplace = "/data/temp" #workplace directory
logdir = os.path.join(workplace, 'logs')
if not os.path.isdir(logdir):
    os.mkdir(logdir)
errorfile = os.path.join(ieo.logdir, 'FLARES_BurnScar_Errors.log')

tiles = ieo.gettilelist()

tfile = os.path.join(workplace, 'sentinel2', 'transferred_tiles.csv')
badfile = os.path.join(workplace, 'sentinel2', 'bad_tiles.csv')
processedtiles = []
for f in [tfile, badfile]:
    if os.path.isfile(f):
        with open(f, 'r') as lines:
            for line in lines:
                line = line.strip().split(',')
                for x in line:
                    if not x in processedtiles:
                        processedtiles.append(x)

if len(processedtiles) > 0:
    print(f'{len(processedtiles)} have already be processed. Removing any processed tiles from processing list.')
    while any(tile in tiles for tile in processedtiles):
        for tile in tiles:
            if tile in processedtiles:
                print(f'Tile {tile} has already been processed, skipping.')
                tiles.pop(tiles.index(tile))

for tile in tiles:
    now = datetime.datetime.now()
    print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} Processing tile: {tile} ({tiles.index(tile) + 1}/{len(tiles)}) ')
    logfile = os.path.join(logdir, f'{tile}_{now.strftime("%Y%m%d-%H%M%S")}.txt')
    with open(logfile, 'w') as fp:
        x = subprocess.run(['python', 'FLARES_BurnScar.py', '--verbose', '--remove', '--tile', tile], stdout = fp)
    if x.returncode != 0:
        print(f'ERROR: Script failure for tile {tile} returning returncode {x.returncode}. Adding to bad tile list.')
        if not os.path.isfile(badfile):
            with open(badfile, 'w') as output:
                output.write(tile)
        else: 
            with open(badfile, 'a') as output:
                output.write(f',{tile}')
        
print('Processing complete.')
            