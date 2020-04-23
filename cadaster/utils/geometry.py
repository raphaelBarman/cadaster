import numpy as np
from shapely import geometry


def fix_geom(geom):
    if geom.is_valid:
        return geom
    else:
        return geom.buffer(0)


def to_multipoly(poly):
    if poly.geom_type != "MultiPolygon":
        poly = geometry.MultiPolygon([poly])
    return poly


def int_coords(x):
    return np.array(x).round().astype(np.int32)
