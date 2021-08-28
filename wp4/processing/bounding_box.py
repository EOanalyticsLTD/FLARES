"""Some functions to calculate a bounding box for a given lat long coordinate, found on stackoverflow:
https://stackoverflow.com/questions/238260/how-to-calculate-the-bounding-box-for-a-given-lat-lng-location

Not the most accurate but ok for the CAMS product with a relatively low resolution"""

import math

# Semi-axes of WGS-84 geoidal reference
WGS84_A = 6378137.0  # Major semiaxis [m]
WGS84_B = 6356752.3  # Minor semiaxis [m]


def _deg2rad(degrees):
    """degrees to radians"""
    return math.pi*degrees/180.0


def _rad2deg(radians):
    """radians to degrees"""
    return 180.0*radians/math.pi


def wgs84_earth_radius(lat):
    """Earth radius at a given latitude, according to the WGS-84 ellipsoid [m].
     http://en.wikipedia.org/wiki/Earth_radius"""

    An = WGS84_A*WGS84_A * math.cos(lat)
    Bn = WGS84_B*WGS84_B * math.sin(lat)
    Ad = WGS84_A * math.cos(lat)
    Bd = WGS84_B * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )


def bounding_box(latitude, longitude, halfside_km):
    """
    # Bounding box surrounding the point at given coordinates, assuming local approximation of Earth surface as a sphere
     of radius given by WGS84.

    :param latitude:
    :param longitude:
    :param halfside_km:
    :return:
    """
    lat = _deg2rad(latitude)
    lon = _deg2rad(longitude)
    half_side = 1000*halfside_km

    # Radius of Earth at given latitude
    radius = wgs84_earth_radius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    lat_min = lat - half_side/radius
    lat_max = lat + half_side/radius
    lon_min = lon - half_side/pradius
    lon_max = lon + half_side/pradius

    return (_rad2deg(lat_min), _rad2deg(lon_min), _rad2deg(lat_max), _rad2deg(lon_max))


def mask_bounding_box(ds, latitude, longitude, halfside_km):
    """Crop out the area within a bounding box for a coordinate in a NetCDF file"""
    bb = bounding_box(latitude, longitude, halfside_km)

    # select min and max longitude and latitude to select the cams data within the bounding box
    min_lon = 360 + bb[1]
    min_lat = bb[0]
    max_lon = 360 + bb[3]
    max_lat = bb[2]

    mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
    mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    ds = ds.where(mask_lon & mask_lat, drop=True)

    return ds


def apply_mask(ds, min_latitude, max_latitude, min_longitude, max_longitude):
    """
    Apply a mask to a xarray dataset based on min/max lat long coordinates

    :param ds: xarray dataset to apply the mask to
    :param min_latitude: min latitude
    :param max_latitude: max latitude
    :param min_longitude: min longitude
    :param max_longitude: max longitude
    :return: masked xarray dataset
    """

    mask_lon = (ds.longitude >= min_longitude) & (ds.longitude <= max_longitude)
    mask_lat = (ds.latitude >= min_latitude) & (ds.latitude <= max_latitude)
    ds = ds.where(mask_lon & mask_lat, drop=True)

    return ds