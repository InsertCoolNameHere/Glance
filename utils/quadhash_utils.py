#import geohash2
import mercantile
import random

#print(geohash2.encode(42.6, -5.6))

# GET QUADHASH TILE OF A GIVEN COORDINATE
def get_quad_tile(lat, lon, precision):
    ret = mercantile.tile(lon,lat,precision)
    return ret

def get_quad_key_from_tile(x, y, zoom):
    return mercantile.quadkey(x, y, zoom)

# GIVEN A QUAD_KEY, GET THE CORRESPONDING QUAD TILE
def get_tile_from_key(key):
    return mercantile.quadkey_to_tile(key)


# GET QUADHASH STRING OF A GIVEN COORDINATE
def get_quad_key(lat, lon, zoom):
    tile = get_quad_tile(lat, lon, precision=zoom)
    print(tile)
    return get_quad_key_from_tile(tile.x, tile.y, tile.z)

#GIVEN A ZOOM LEVEL, WHAT IS THE MAX POSSIBLE TILE NUMBER HERE?
def get_max_possible_xy(zoom):
    if zoom == 0:
        return 0
    return 2**zoom-1


# GIVEN A TILE, VERIFY IT IS VALID
def validate_tile(tile):
    max_xy = get_max_possible_xy(tile.z)

    if tile.x > max_xy or tile.x < 0 or tile.y > max_xy or tile.y < 0:
        return False

    return True


# GIVEN A BOX, FIND ALL TILES THAT LIE INSIDE THAT COORDINATE BOX
def find_all_inside_box(lat1, lat2, lon1, lon2, zoom):
    all_tiles = []
    top_left_quad_tile = get_quad_tile(lat2, lon1, zoom)
    bottom_right_quad_tile = get_quad_tile(lat1, lon2, zoom)

    print("TOP_LEFT & BOTTOM_RIGHT: ",top_left_quad_tile, bottom_right_quad_tile)

    x1 = top_left_quad_tile.x
    x2 = bottom_right_quad_tile.x

    y1 = top_left_quad_tile.y
    y2 = bottom_right_quad_tile.y

    for i in range(x1, x2+1):
        for j in range(y1,y2+1):
            all_tiles.append(mercantile.Tile(x=i,y=j,z=zoom))

    return all_tiles

#GIVEN A TILE, FIND THE SMALLER TILES THAT LIE INSIDE
def get_inner_tiles(tile_string):
    combos = range(4)

    children = []

    for i in combos:
        t_s = tile_string+str(i)
        children.append(get_tile_from_key(t_s))
    return children


#GIVEN A QUAD_TILE, GET ITS LAT-LNG BOUNDS
def get_bounding_lng_lat(tile_key):
    tile = get_tile_from_key(tile_key)
    bounds = mercantile.bounds(tile)
    #print(tile_key, tile, bounds)
    return (bounds.north, bounds.south, bounds.east, bounds.west)



if __name__ == '__main__':
    tile_key = "02132333222"

    print("BOUNDS",get_bounding_lng_lat(tile_key))


