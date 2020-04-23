from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cadaster.segmentation.mask.config import Annotation, ClassLabel, mask_config
from cadaster.segmentation.mask.mask import get_mask_binary
from cadaster.utils.image import hex2rgb, unique_rows

#
# @mask_config.capture
# def get_labels_and_colors(
#     annotation_data: List[Annotation],
#     classes: List[ClassLabel],
#     classes_offset: int,
#     num_classes: int,
#     nodes: Dict[str, object],
#     edges: Dict[str, object],
#     multilabel: bool,
#     no_class_overlap: bool,
#     mask_cache: Optional[Dict[Annotation, np.array]] = None,
#     progress: bool = False,
# ):
#     all_colors = gather_colors(nodes, edges, classes)
#
#     if multilabel:
#         label_factory = LabelsFactory()
#         for annotation in annotation_data:
#             mask = get_mask_binary(
#                 annotation,
#                 classes,
#                 classes_offset,
#                 num_classes,
#                 nodes,
#                 edges,
#                 multilabel,
#                 no_class_overlap,
#                 progress,
#             )
#             label_factory.add_mask(mask, num_classes, multilabel)
#             if mask_cache is not None:
#                 mask_cache[annotation] = mask
#         labels = label_factory.labels
#     else:
#         labels = np.array(range(num_classes + 1))
#
#     labels_colors = get_labels_colors(labels, all_colors, multilabel)
#
#     return labels, labels_colors


@mask_config.capture
def all_possible_labels_colors(
    nodes: Dict[str, object],
    edges: Dict[str, object],
    classes: List[ClassLabel],
    num_classes: int,
    multilabel: bool,
):
    all_colors = gather_colors(nodes, edges, classes)
    if not multilabel:
        labels = np.array(range(num_classes + 1))
        labels_colors = np.array([[0, 0, 0]] + all_colors.tolist())
    else:
        labels_colors = []
        # labels = sort_labels(np.array(list(product([0, 1], repeat=num_classes))))
        labels = (
            (np.arange(0, 2 ** num_classes)[:, None] & (1 << np.arange(num_classes)))
            > 0
        ).astype(int)
        for label in labels:
            if label.sum() == 0:
                color = np.array([0, 0, 0])
            else:
                color = np.round(
                    np.mean(all_colors[np.where(label)[0]], axis=0)
                ).astype(int)
            labels_colors.append(color)
        labels_colors = np.array(labels_colors)
    return labels, labels_colors


def gather_colors(
    nodes: Dict[str, object],
    edges: Dict[str, object],
    classes: List[ClassLabel],
    add_empty: bool = False,
) -> np.array:
    all_colors = []
    if add_empty:
        all_colors.append("#000000")
    if edges["use"]:
        all_colors.append(edges["color"])
    if nodes["use"]:
        all_colors.append(nodes["color"])
    for class_ in classes:
        all_colors.append(class_["color"])
    all_colors = np.array([hex2rgb(color) for color in all_colors])

    return all_colors


# class LabelsFactory:
#     def __init__(self):
#         self._all_labels = []
#
#     def add_mask(self, mask_binary: np.array, num_classes: int, multilabel: bool):
#         self._all_labels.append(mask_binary_to_labels(mask_binary, num_classes, multilabel))
#
#     @property
#     def labels(self):
#         return merge_labels(self._all_labels)


def get_labels_colors(
    labels: np.array, all_colors: np.array, multilabel: bool
) -> np.array:
    labels_colors = []
    for label in labels:
        if multilabel:
            if label.sum() == 0:
                color = np.array([0, 0, 0])
            else:
                color = np.round(
                    np.mean(all_colors[np.where(label)[0]], axis=0)
                ).astype(int)
        else:
            if label == 0:
                color = np.array([0, 0, 0])
            else:
                color = all_colors[label - 1]
        labels_colors.append(color)
    labels_colors = np.array(labels_colors)
    return labels_colors


def mask_binary_to_labels(mask: np.array, num_classes: int, multilabel: bool):
    if multilabel:
        labels = sort_labels(unique_rows(mask))
    else:
        labels = np.array(range(num_classes + 1))
    return labels


#
# def merge_labels(labels: List[np.array]):
#     return sort_labels(unique_rows(np.concatenate(labels)))


# def sort_labels(labels):
#     num_classes = labels.shape[1]
#     labels = pd.DataFrame(labels)
#     labels["sum"] = labels.sum(axis=1)
#     labels = (
#         labels.sort_values(by=["sum"] + list(range(num_classes - 1, -1, -1)))
#         .iloc[:, :num_classes]
#         .values
#     )
#     return labels


def sort_labels(labels, labels_colors):
    num_classes = labels.shape[1]
    labels_df = pd.DataFrame(np.concatenate([labels, labels_colors], axis=-1))
    labels_df["sum"] = labels_df.iloc[:, :num_classes].sum(axis=1)
    labels_df = labels_df.sort_values(by=["sum"] + list(range(num_classes - 1, -1, -1)))
    labels = labels_df.iloc[:, :num_classes].values
    labels_colors = labels_df.iloc[:, num_classes:-1].values
    return labels, labels_colors
