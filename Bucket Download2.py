# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:59:31 2021
​
@author: guyse
"""
import os, argparse, sys, glob, boto3, datetime
from subprocess import Popen
from pkg_resources import resource_stream, resource_string, resource_filename, Requirement

parser = argparse.ArgumentParser(description = 'IEO Object Storage interface submodule.')
# parser.add_argument('--indir', type = str, default = None, help = 'Optional input directory in which to search for files. This is ignored if --batch=True.')
# parser.add_argument('--outdir', type = str, default = None, help = 'Optional output directory in which to save products.')
# parser.add_argument('--baseoutdir', type = str, default = config['DEFAULT']['BASEOUTPUTDIR'], help = 'Base output directory in which to save products.')
# parser.add_argument('-o', '--outfile', type = str, default = None, help = 'Output product filename. If --infile is not set, then this flag will be ignored.')
# parser.add_argument('-i', '--infile', type = str, default = None, help = 'Input file name.')
# parser.add_argument('-g', '--graph', type = str, default = r'C:\Users\Guy\.snap\graphs\User Graphs\MCI_Resample_S2_20m.xml', help = 'ESA SNAP XML graph file path.')
# parser.add_argument('-e', '--op', type = str, default = None, help = 'ESA SNAP operator.')
# parser.add_argument('-p', '--properties', type = str, default = r'D:\Imagery\Scripts\Mci.S2.properties', help = 'ESA SNAP GPT command properties text file path.')
# parser.add_argument('-d', '--dimap', type = bool, default = 'store_true', help = 'Input files are BEAM DIMAP.')
# parser.add_argument('--gpt', type = str, default = r'C:\Program Files\snap\bin\gpt', help = 'ESA SNAP XML graph file path.')
# parser.add_argument('--overwrite', type = bool, default = False, help = 'Overwrite any existing files.')
# parser.add_argument('--mgrs', type = str, default = '30TUK', help = 'Sentinel-2 MGRS Tile name.')
# parser.add_argument('--batch', type = bool, default = True, help = 'Process all available scenes for a given satellite/ sensor combination.')
# parser.add_argument('--sentinel', type = str, default = '2', help = 'Sentinel satellite number (default = 2).')
# parser.add_argument('--product', type = str, default = 'l2a', help = 'Sentinel product type (default = l2a, will be different for different sensors).')
# parser.add_argument('--bucket', type = str, default = None, help = 'Only process data from a specific bucket.')
parser.add_argument('--url', type = str, default = 'https://obs.eu-de.otc.t-systems.com', help = 'Alternative S3 bucket URL. If used, you must also present a different --credentials file from the default.')
# parser.add_argument('--credentials', type = str, default = config['S3']['credentials'], help = 'Full path of S3 credentials CSV file to use.')
# parser.add_argument('--warp', type = str, default = None, help = 'Warp products to specific projection EPSG code. Example: for Irish Transverse Mercator (EPSG:2157), use "2157". Not implemented yet.')

args = parser.parse_args()
url = args.url
aws_id = 'NB808KHW1WXALYHUOVKV'
aws_secret = 'wyyq99J9YGSMkIjSLVUJV8TYXqh43B32z7iQz6bL'


def readcredentials():
    credentials = {}
    with open(args.credentials, 'r') as lines:
        for line in lines:
            line = line.rstrip().split(',')
            if line[0] == 'User Name':
                headers = line
            else:
                if len(line) > 0:
                    for i in range(len(line)):
                        credentials[headers[i]] = line[i]
    return credentials

def s3resource(url, *args, **kwargs):
    s3res = boto3.resource('s3',aws_access_key_id = aws_id , aws_secret_access_key = aws_secret, endpoint_url = url)
    return s3res

def s3client(url, **kwargs):
    s3cli = boto3.client('s3',  aws_access_key_id = aws_id , aws_secret_access_key = aws_secret, endpoint_url = url)
    return s3cli

def movefile(bucket, fin, fout, *args, **kwargs):
    # This function will more/ rename a local file within a specified S3 bucket
    # bucket = kwargs.get('bucket', None) # name of S3 bucket to save files.
    # filename = kwargs.get('filename', None) # single file to be coped to S3 bucket
    # filelist = kwargs.get('filelist', None) # list of files to be copied to S3 bucket
    # copydir = kwargs.get('copydir', None) # directory containing files to be copied to S3 bucket
    # inbasedir = kwargs.get('inbasedir', None) # start of local directory path to be stripped from full file path. Only used if "copydir" is is not used.
    # targetdir = kwargs.get('targetdir', None) # name of directory in which to copy file or files. If empty, will be taken from the directory of the first file in the list if "copydir" is used, or be determined by processing time
    remove = kwargs.get('remove', True) # If set to true, this deletes the local copy of the file
    print(f'Copying {fin} to {fout}')
    s3res.Object(bucket, fout).copy_from(CopySource = '{}/{}'.format(bucket, fin))
    if remove:
        print(f'Removing {fin} from bucket {bucket}.')
        s3res.Object(bucket, fin).delete()
        
def renametile(bucket, prefix, f):
    i = f.find('.')
    parts = f[:i].split('_')
    datetuple = datetime.datetime.strptime(parts[1], '%Y%j')
    year = parts[1][:4]
    newbasename = '{}_{}_{}.{}'.format(parts[0], datetuple.strftime('%Y%m%d'), parts[2], f[i + 1:])
    fout = '{}/{}/{}/{}'.format(prefix, parts[2], year, newbasename)
    fin = '{}/{}'.format(prefix, f)
    movefile(bucket, fin, fout)

def copyfilesinbucket(*args, **kwargs):
    # This function will copy local files to a specified S3 bucket
    bucket = kwargs.get('bucket', None) # name of S3 bucket to save files.
    filename = kwargs.get('filename', None) # single file to be coped to S3 bucket
    filelist = kwargs.get('filelist', None) # list of files to be copied to S3 bucket
    copydir = kwargs.get('copydir', None) # directory containing files to be copied to S3 bucket
    inbasedir = kwargs.get('inbasedir', None) # start of local directory path to be stripped from full file path. Only used if "copydir" is is not used.
    targetdir = kwargs.get('targetdir', None) # name of directory in which to copy file or files. If empty, will be taken from the directory of the first file in the list if "copydir" is used, or be determined by processing time
    remove = kwargs.get('remove', False) # If set to true, this deletes the local copy of the file
    flist = []
    dirlist = []
    i = 0
    if copydir:
        i = len(copydir)
        if not os.path.isdir(copydir):
            print('ERROR: The path {} does not exist or is not a folder. Exiting.'.format(copydir))
            sys.exit()
        for root, dirs, files in os.walk(copydir):
            for name in files:
                flist.append(os.path.join(root, name))
    elif filelist:
        try:
            if (len(filelist) == 0) or (not isinstance(filelist, list)):
                print('ERROR: "filelist" keyword used, but either has zero items or is not a list object. Exiting.')
                sys.exit()
        except Exception as e:
            print('ERROR, an exception has occurred, exiting: {}'.format(e))
            sys.exit()
        flist = filelist
    else:
        if not os.path.isfile(filename):
            print('ERROR: {} does not exist. Exiting.'.format(filename))
            sys.exit()
        else:
            flist.append(filename)
    if not targetdir:
        if copydir:
            if flist[0][i : i + 1] == '/':
                i += 1
            diritems = flist[0][i:].split('/') 
            if len(diritems) == 1:
                now = datetime.datetime.now()
                targetdir = now.strftime('%Y%m%d-%H$M%S')
            else:
                targetdir = diritems[0]
        else:
            now = datetime.datetime.now()
            targetdir = now.strftime('%Y%m%d-%H$M%S')
    if inbasedir and not copydir:
        if os.path.isdir(inbasedir):
            if inbasedir.endswith('/'):
                i = len(inbasedir)
            else:
                i = len(inbasedir) + 1
        else:
            print('ERROR: "inbasedir" {} is not a folder on the local machine. Files will be saved to the base target directory {}.'.format(inbasedir, targetdir))
    numerrors = 0
    for f in flist:
        print('Now copying {} to bucket {}. ({}/{})'.format(f, bucket, flist.index(f) + 1, len(flist)))
        if not os.path.isfile(f):
            print('ERROR: {} does not exist on disk, skipping.'. format(f))
            numerrors += 1
        else:
            if i > 0:
                targetfile = "{}/{}".format(targetdir, f[i:])
            else:
                targetfile = "{}/{}".format(targetdir, os.path.basename(f))
            s3cli.upload_file(f, bucket, targetfile)
            if remove:
                print('Deleting local file: {}'.format(f))
                os.remove(f)
    print('Upload complete. {}/{} files uploaded, with {} errors.'. format(len(flist) - numerrors, len(flist), numerrors))
            
            
def downloadfile(outdir, s3_object):
# def downloadfile(outdir, bucket, s3_object):
    print('Downloading file {} from bucket {} to: {}'.format(s3_object, bucket, outdir))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, s3_object)
    # s3cli.download_file(bucket, s3_object, outfile)
    s3res.meta.client.download_file('landsat', s3_object, outfile)
    
def getbucketobjects(bucket, prefix):
    # code borrowed from 
    print('Retrieving objects for S3 bucket: {}'.format(bucket))
    outdict = {}
    try:
        contents = s3cli.list_objects_v2(Bucket = bucket, Prefix = prefix)['Contents']
        for s3_key in contents:
            s3_object = s3_key['Key']
            if not s3_object.endswith('/'):
                outdname, outfname = os.path.split(s3_object)
                if outdname:
                    if not outdname in outdict.keys():
                        outdict[outdname] = []
                    outdict[outdname].append(outfname)
                
            else:
                outdict[s3_object[:-1]] = []
        return outdict
    except:
        print('Error: unable to get directory listing.')
        return None

def downloadscene(scenedict, sceneid, downloaddir):
    # code borrowed from 
    outdir = os.path.join(downloaddir, sceneid)
    print('Downloading scene {} to: {}'.format(sceneid, outdir))
    if not os.path.isdir(outdir):
        os.path.mkdir(outdir)
    bucket = scenedict[sceneid]['bucket']
    prefix = scenedict[sceneid]['prefix']
    i = len(prefix)
    contents = s3cli.list_objects_v2(Bucket = bucket, Prefix = prefix)['Contents']
    for s3_key in contents:
        s3_object = s3_key['Key']
        if not s3_object.endswith('/'):
            outdname, outfname = os.path.split(s3_object[i + 1:])
            outdir = os.path.join(outdir, outdname)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, outfname)
            s3cli.download_file(bucket, s3_object, outfile)
        else:
            if not os.path.isdir(s3_object):
                os.makedirs(s3_object)
    print('Scene {} has been downloaded.'.format(sceneid))

s3cli = s3client(args.url)
s3res = s3resource(args.url)         
basedir = '/data/ieo-data'
buckets = ['landsat']
bucketdirs = ['SR', 'ST', 'EVI', 'NDVI', 'pixel_qa', 'radsat_qa', 'aerosol_qa']
landsats = ['LC8', 'LE7', 'LT5']

startyear = 2015
endyear = 2021

# for bucket in buckets:
#     print('Now transfering files for bucket {}.'.format(bucket))
#     # if bucket == 'ingested':
#     #     filelist = glob.glob(os.path.join(basedir, bucket, '*.tar'))
#     #     if len(filelist) > 0:
#     #         print('Found {} files to transfer.'.format(len(filelist)))
#     #         copyfilestobucket(bucket = bucket, filelist = filelist, targetdir = 'landsat', remove = True)
#     # else:
#     for bucketdir in bucketdirs:
#         y = startyear
#         while y <= endyear:
#             if y < 2019:
#                 landsats = ['LE7']
#             else:
#                 landsats = ['LC8', 'LE7']
#             for landsat in landsats:
#                 sprefix = f'{bucketdir}/{landsat}_{y}'
#                 print('Now searching for objects starting with {}'.format(sprefix))
#                 d = getbucketobjects(bucket, sprefix)
#                 if isinstance(d, dict):
#                     if len(d[bucketdir]) > 0:
#                         print('Found {} objects to rename.'.format(len(d[bucketdir])))
#                         for f in d[bucketdir]:
#                             renametile(bucket, bucketdir, f)
#             y += 1
#         # filelist = glob.glob(os.path.join(basedir, bucket, bucketdir, '*.*'))
#         # if len(filelist) > 0:
#         #     print('Found {} files to transfer in folder {}.'.format(len(filelist), bucketdir))
#         #     copyfilestobucket(bucket = bucket, filelist = filelist, targetdir = bucketdir, remove = True)


for bucket in buckets:
    if bucket == 'landsat':

        for bucketdir in bucketdirs:
            print(bucketdir)
            y = startyear
            if bucketdir == 'SR':
                while y <= endyear:
                    print('\n\n\n\n' + str(y))
                    # if y < 2019:
                    #     landsats = ['LE7']
                    # else:
                    #     landsats = ['LC8', 'LE7']
                    landsats = ['LC8', 'LE7']
                    for landsat in landsats:
                        sprefix = f'{bucketdir}/{landsat}_{y}'
                        SR_bucket = getbucketobjects(bucket,sprefix)
                        try:
                            images = SR_bucket.values()
                        except:
                            break
                        for item in images:
                            for image in item:
                                if 'K07' in image:
                                    downloadfile(r'C:\Users\Zam\Desktop\Masters\EOanalytics', 'SR/' + image)
                                    print(image)

                    y += 1
                
                    
                    # filelist = glob.glob(os.path.join(basedir, bucket, bucketdir, '*.*'))
                    # print(len(filelist))
                    # if len(filelist) > 0:
                    #     print('Found {} files in folder {}.'.format(len(filelist), bucketdir))
                
print('Processing complete.')