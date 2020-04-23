from functools import lru_cache
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from cadaster.gis.osm import read_osm
from cadaster.gis.raster import Raster
from cadaster.segmentation.mask.config import (
    Annotation,
    ClassLabel,
    mask_config,
)
from cadaster.segmentation.mask.drawing import MaskDrawer


def mask_binary_to_labels(mask: np.array, num_classes: int, multilabel: bool):
    if multilabel:
        labels = sort_labels(unique_rows(mask))
    else:
        labels = np.array(range(num_classes + 1))
    return labels


def merge_labels(labels: List[np.array]):
    return sort_labels(unique_rows(np.concatenate(labels)))


def unique_rows(array):
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    array = np.ascontiguousarray(array)
    array_row_view = array.view(
        np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    )
    _, unique_row_indices = np.unique(array_row_view, return_index=True)
    return array[unique_row_indices]


def sort_labels(labels):
    num_classes = labels.shape[1]
    labels = pd.DataFrame(labels)
    labels["sum"] = labels.sum(axis=1)
    labels = (
        labels.sort_values(by=["sum"] + list(range(num_classes - 1, -1, -1)))
        .iloc[:, :num_classes]
        .values
    )
    return labels


@mask_config.capture
def mask_binary_to_color(
    mask: np.array, labels_colors: np.array, num_classes: int, multilabel: bool,
) -> (np.array, np.array):
    if multilabel:
        indices = mask.dot(1 << np.arange(num_classes))
    else:
        mask_flat = mask.reshape(-1, num_classes)
        indices = (mask_flat.argmax(axis=1) + 1) - (mask_flat.max(axis=1) == 0)
        indices = indices.reshape(mask.shape[0], mask.shape[1])
    mask_colors = np.take(labels_colors, indices, axis=0).astype(np.uint8)
    return mask_colors, indices


# @lru_cache(maxsize=3)
@mask_config.capture
def get_mask_binary(
    annotation: Annotation,
    classes: List[ClassLabel],
    classes_offset: int,
    num_classes: int,
    nodes: Dict[str, object],
    edges: Dict[str, object],
    multilabel: bool,
    no_class_overlap: bool,
    progress: bool,
) -> np.array:
    raster_size, osm_data = load_data(annotation)
    mask = np.zeros((raster_size[0], raster_size[1], num_classes), dtype=np.int32)

    mask_drawer = MaskDrawer(
        osm_data,
        classes,
        edges,
        nodes,
        num_classes,
        classes_offset,
        multilabel,
        no_class_overlap,
        progress,
    )

    mask = mask_drawer.draw(mask)

    return mask


def load_data(annotation: Annotation) -> (Tuple[int], pd.DataFrame):
    raster = Raster(annotation.raster)
    osm_data = read_osm(annotation.osm, raster.map2raster)

    return (raster.size.height, raster.size.width), osm_data
