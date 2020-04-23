from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

import cv2
import numpy as np
import pandas as pd
from shapely import geometry
from tqdm import tqdm

from cadaster.segmentation.mask.config import ClassLabel
from cadaster.utils.geometry import int_coords


@dataclass
class MaskDrawer:
    osm_data: pd.DataFrame
    classes: List[ClassLabel]
    edges: Dict[str, object]
    nodes: Dict[str, object]
    num_classes: int
    classes_offset: int
    multilabel: bool
    no_class_overlap: bool
    progress: bool = False

    def draw(self, mask: np.array, inplace: bool = False) -> np.array:
        if not inplace:
            mask = mask.copy()
        mask = self.draw_classes(mask, True)
        if self.edges["use"]:
            mask = self.draw_edges(mask, True)
        if self.nodes["use"]:
            mask = self.draw_nodes(mask, True)
        return mask

    def draw_classes(self, mask: np.array, inplace: bool = False):
        if not inplace:
            mask = mask.copy()
        for class_idx, class_ in enumerate(
            tqdm(self.classes, disable=not self.progress)
        ):
            mask = self.draw_class(class_, class_idx + self.classes_offset, mask)
        return mask

    def draw_class(
        self,
        class_: ClassLabel,
        class_index: int,
        mask: np.array,
        inplace: bool = False,
    ):
        if not inplace:
            mask = mask.copy()
        polys = self.osm_data.loc[filter2osm_mask(class_["filter"], self.osm_data)][
            "geometry"
        ].values
        poly_mask = polys2mask(polys, mask.shape[:2])
        if self.multilabel and self.no_class_overlap:
            mask[poly_mask] = [
                1 if i == class_index else 0 for i in range(self.num_classes)
            ]
        else:
            mask[poly_mask, class_index] = 1
        return mask

    def draw_edges(self, mask: np.array, inplace: bool = False):
        if not inplace:
            mask = mask.copy()
        edges_mask = polys2contour_mask(
            self.osm_data["geometry"].values, mask.shape[:2], self.edges["width"]
        )
        if self.multilabel:
            mask[edges_mask, 0] = 1
        else:
            mask[edges_mask] = [1 if i == 0 else 0 for i in range(self.num_classes)]
        return mask

    def draw_nodes(self, mask: np.array, inplace: bool = False):
        if not inplace:
            mask = mask.copy()
        nodes_mask = polys2nodes_mask(
            self.osm_data["geometry"].values, mask.shape[:2], self.nodes["radius"]
        )
        if self.multilabel:
            mask[nodes_mask, self.classes_offset - 1] = 1
        else:
            mask[nodes_mask] = [
                1 if i == self.classes_offset - 1 else 0
                for i in range(self.num_classes)
            ]
        return mask


def polys2mask(polys: List[geometry.MultiPolygon], im_size: Tuple[int]):
    img_mask = np.zeros(im_size, dtype=np.uint8)
    for polygons in polys:
        internal_mask = np.zeros(im_size, np.uint8)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [
            int_coords(pi.coords) for poly in polygons for pi in poly.interiors
        ]
        cv2.fillPoly(internal_mask, exteriors, 1)
        cv2.fillPoly(internal_mask, interiors, 0)
        img_mask = img_mask | internal_mask
    return img_mask.astype(bool)


def polys2contour_mask(
    polys: List[geometry.MultiPolygon], im_size: Tuple[int], thickness: int = 3
):
    img_mask = np.zeros(im_size, dtype=np.uint8)
    for polygons in polys:
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [
            int_coords(pi.coords) for poly in polygons for pi in poly.interiors
        ]
        cv2.drawContours(img_mask, exteriors, -1, 1, thickness)
        cv2.drawContours(img_mask, interiors, -1, 1, thickness)
    return img_mask.astype(bool)


def polys2nodes_mask(
    polys: List[geometry.MultiPolygon], im_size: Tuple[int], radius: int = 3
):
    img_mask = np.zeros(im_size, dtype=np.uint8)
    for polygons in polys:
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [
            int_coords(pi.coords) for poly in polygons for pi in poly.interiors
        ]
        coords = [tuple(coord) for poly in exteriors + interiors for coord in poly]
        for coord in coords:
            cv2.circle(img_mask, coord, radius, 1, -1)
    return img_mask.astype(bool)


def filter2osm_mask(filter_: Union[str, Dict[str, str]], osm_data: pd.DataFrame):
    if isinstance(filter_, str):

        def filter_func(tags):
            return filter_ in tags

    else:

        def filter_func(tags):
            for key, value in filter_.items():
                if key not in tags or tags[key] != value:
                    return False
            return True

    return osm_data["tags"].apply(filter_func)
