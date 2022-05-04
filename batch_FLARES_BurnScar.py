# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:38:32 2022

@author: guyse
"""

import os, sys, subprocess, datetime, shutil, glob, argparse
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

parser = argparse.ArgumentParser('This script batch processes FLARES_BurnScar.py.')
parser.add_argument('--starttile', type = str, default = None, help = 'Starting tile')
parser.add_argument('--endtile', type = str, default = None, help = 'Ending tile')
parser.add_argument('--version', type = int, default = 2, help = 'Version to use. Default = 2')
args = parser.parse_args()

if args.version == 1:
    burnscript = 'FLARES_BurnScar.py'
else:
    burnscript = 'FLARES_BurnScar_V2.py'

workplace = "/data/temp" #workplace directory
logdir = os.path.join(workplace, 'logs')
if not os.path.isdir(logdir):
    os.mkdir(logdir)
errorfile = os.path.join(ieo.logdir, 'FLARES_BurnScar_Errors.log')
maxtries = 5
tiles = ieo.gettilelist()

if args.starttile or args.enddtile:
    if args.starttile:
        i = tiles.index(args.starttile)
    else:
        i = 0
    if args.endtile:
        j = tiles.index(args.endtile)
    else:
        j = len(tiles) - 1
    tiles = tiles[i:j]

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
    tries = 1
    while tries <= maxtries:
        now = datetime.datetime.now()
        print(f'{now.strftime("%Y-%m-%d %H:%M:%S")} Processing tile: {tile} ({tiles.index(tile) + 1}/{len(tiles)}), attempt {tries}/{maxtries}.')
        logfile = os.path.join(logdir, f'{tile}_{now.strftime("%Y%m%d-%H%M%S")}.txt')
        
        with open(logfile, 'w') as fp:
            x = subprocess.run(['python', burnscript, '--verbose', '--remove', '--tile', tile], stdout = fp)
        if x.returncode != 0:
            
            if tries == maxtries:
                print(f'ERROR: Script failure for tile {tile} returning returncode {x.returncode}. Adding to bad tile list.')
                if not os.path.isfile(badfile):
                    with open(badfile, 'w') as output:
                        output.write(tile)
                    
                else: 
                    with open(badfile, 'a') as output:
                        output.write(f',{tile}')
                for d in ['sentinel2', 'landsat']:
                    dirname = os.path.join(workplace, d, tile)
                    if os.path.isdir(dirname):
                        print(f'Deleting output directory and files for tile {tile}.')
                        shutil.rmtree(dirname)
                for d in [ieo.Sen2srdir, ieo.srdir]:
                    flist = glob.glob(os.path.join(d, f'*_{tile}.*'))
                    if len(flist) > 0:
                        for f in flist:
                            if not '_2022' in f:
                                print(f'Deleting file: {f}')
                                os.remove(f)
            else: 
                print('ERROR running script. Attempting again.')
                tries += 1
        else:
            break
        
print('Processing complete.')
            