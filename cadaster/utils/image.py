from typing import NamedTuple, Union

import cv2
import numpy as np

ImageSize = NamedTuple("ImageSize", [("height", int), ("width", int)])


def hex2rgb(hex_color: str):
    return tuple([int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)])


def unique_rows(array):
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    array = np.ascontiguousarray(array)
    array_row_view = array.view(
        np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    )
    _, unique_row_indices = np.unique(array_row_view, return_index=True)
    return array[unique_row_indices]


def get_image_colors(image: Union[str, np.array]):
    if isinstance(image, str):
        image = cv2.imread(image)[:, :, ::-1]
    return unique_rows(image)
