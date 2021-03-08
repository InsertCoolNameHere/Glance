import gdal
from affine import Affine
from osgeo import osr
from utils.quadhash_utils import *

def retrieve_pixel_value(x,y,data_source):
    #print(data_source.GetProjection())
    #(data_source.GetGeoTransform())
    forward_transform = Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    return pixel_coord

def retrive_coord_from_pixel(x, y, data_source):
    forward_transform = Affine.from_gdal(*data_source.GetGeoTransform())
    px, py = forward_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    return pixel_coord

#RETURNS LAT-LON BOUNDS OF A GEOHASH
def decode_geohash(geocode):
    lat, lon, l1, l2 = geohash2.decode_exactly(geocode)
    return lat-l1, lat+l1, lon-l2, lon+l2

def get_gdal_obj(filename):
    return gdal.Open(filename)

# GIVEN A SINGLE EPSG LAT-LON CONVERT TO REGULAR SYSTEM
def convert_EPSG_to_latlon(src_lat, src_lon, dataset):
    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjection())

    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    # Create the transform - this can be used repeatedly
    transform = osr.CoordinateTransformation(source, target)

    lon, lat, z = transform.TransformPoint(src_lat, src_lon)
    return lat, lon

def get_pixel_from_lat_lon(latlons, dataset):
    #converting coordinate systems
    # Setup the source projection
    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjection())
    #print(source)
    # The target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    # Create the transform - this can be used repeatedly
    transform = osr.CoordinateTransformation(target, source)

    pixels=[]
    for lat,lon in latlons:
        x, y, z = transform.TransformPoint(lon, lat)

        ret = retrieve_pixel_value(x, y, dataset)
        #print("TRANSFORMED:", x, y, ret)
        pixels.append(ret)
    return pixels

# CROP OUT A SMALLER GEOHASH FROM A LARGER GEOHASH TILE
def crop_geohash(geocode, datafile):
    lat1, lat2, lon1, lon2 = decode_geohash(geocode)
    return crop_section(lat1, lat2, lon1, lon2, datafile)

# CROPPING A RECTANGLE OUT OF AN IMAGE
def crop_section(lat1, lat2, lon1, lon2, datafile):
    latlons = []
    latlons.append((lat1, lon1))
    latlons.append((lat2, lon1))
    latlons.append((lat2, lon2))
    latlons.append((lat1, lon2))
    return get_pixel_from_lat_lon(latlons, datafile)


# ************************METHOD CALLS

'''
filename = "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRData_Quad/co_test/02310010132_20190604.tif" #path to raster
gdal_obj= get_gdal_obj(filename)
target_hash = "02310010132"
north,south,east,west = get_bounding_lng_lat(target_hash)
print(north,south,east,west)
latlons = []
latlons.append((west, north))
latlons.append((east, north))
latlons.append((east, south))
latlons.append((west, south))

pixels = get_pixel_from_lat_lon(latlons,gdal_obj)

print(pixels)
cropped = crop_irregular_polygon(pixels, filename)
print("Hi")
'''