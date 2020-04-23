from typing import Callable, Optional, Tuple

import geopandas as gpd
import numpy as np
import osmium as osm
import pandas as pd
from shapely import wkb as wkblib, geometry
from shapely.ops import transform

from cadaster.utils.geometry import fix_geom, to_multipoly


def read_osm(
    file: str,
    coords_transformer: Optional[Callable[[Tuple[int, int]], Tuple[int, int]]] = None,
    multipoly: bool = True,
    expand_tags: bool = False,
):
    osm_handler = OSMHandler()
    osm_handler.apply_file(file, locations=True)
    data = osm_handler.data
    data.geometry = data.geometry.apply(fix_geom)
    if coords_transformer is not None:
        data.geometry = data.geometry.apply(lambda g: transform(coords_transformer, g))
    if multipoly:
        data.geometry = data.geometry.apply(to_multipoly)
    if expand_tags:
        data = pd.concat(
            [data[["geometry"]], data.tags.apply(dict).apply(pd.Series)], axis=1
        )
    return data


class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)

        self.data = gpd.GeoDataFrame(columns=["id", "geometry", "tags"])
        self.data.set_index("id", inplace=True)
        self.data.set_geometry("geometry", inplace=True)

        self.wkbfab = osm.geom.WKBFactory()

    def way(self, w):
        self.data.loc[w.id] = {"geometry": self.way2poly(w), "tags": dict(w.tags)}

    def relation(self, r):
        self.data.loc[r.id] = {"geometry": self.relation2poly(r), "tags": dict(r.tags)}
        for member in r.members:
            self.data.drop(member.ref, inplace=True)

    def way2line(self, way):
        wkb = self.wkbfab.create_linestring(way)
        line = wkblib.loads(wkb, hex=True)
        return line

    def line2poly(self, line):
        return geometry.Polygon(list(line.coords))

    def way2poly(self, way):
        return self.line2poly(self.way2line(way))

    def relation2poly(self, relation):
        inners = []
        outers = []

        for member in relation.members:
            poly = self.data.loc[member.ref, "geometry"]
            if member.role == "inner":
                inners.append(poly)
            elif member.role == "outer":
                outers.append(poly)

        if len(outers) == 0:
            return geometry.Polygon()
        elif len(outers) == 1:
            return geometry.Polygon(
                outers[0].exterior.coords,
                holes=[inner.exterior.coords for inner in inners],
            )
        else:
            intersections = np.zeros((len(inners), len(outers)))
            for idx_inner, inner in enumerate(inners):
                for idx_outer, outer in enumerate(outers):
                    intersections[idx_inner][idx_outer] = inner.intersection(outer).area
            max_outers = intersections.argmax(axis=1)
            outers = [(outer, []) for outer in outers]
            for inner, max_outer in zip(inners, max_outers):
                outers[max_outer][1].append(inner)

            polys = []
            for outer, inners in outers:
                polys.append(
                    (outer.exterior.coords, [inner.exterior.coords for inner in inners])
                )
            return geometry.MultiPolygon(polys)
