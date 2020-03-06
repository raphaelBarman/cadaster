from dataclasses import dataclass
from typing import NamedTuple

from shapely import geometry


def fix_geom(p):
    if p.is_valid:
        return p
    else:
        return p.buffer(0)


def to_multipoly(poly):
    if poly.geom_type != 'MultiPolygon':
        poly = geometry.MultiPolygon([poly])
    return poly

ImageSize = NamedTuple("ImageSize", [('height', int), ('width', int)])