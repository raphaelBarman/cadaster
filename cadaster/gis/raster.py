from typing import Union, Optional

import cv2
import pyproj
import rasterio

from cadaster.utils.image import ImageSize


class Raster:
    def __init__(
        self, raster_path: str, raster_crs: Optional[Union[str, int]] = "epsg:3004"
    ):
        self.raster_path = raster_path
        with rasterio.open(raster_path) as raster:
            self.transform = raster.transform
            self.inv_transform = ~self.transform
            if raster_crs is None and raster.crs is not None:
                raster_crs = raster.crs
        self.wsg2crs = pyproj.Transformer.from_crs(
            4326, raster_crs, always_xy=True
        ).transform
        self.crs2wsg = pyproj.Transformer.from_crs(
            raster_crs, 4326, always_xy=True
        ).transform
        self._image = None

    def raster2map(self, x, y):
        return self.coord2coord(x, y, self.transform, self.crs2wsg)

    def map2raster(self, x, y):
        return self.coord2coord(x, y, self.inv_transform, self.wsg2crs)

    # @property
    def image(self, rgb: bool = True):
        if self._image is None:
            self._image = cv2.imread(self.raster_path)[:, :, ::-1]
        if rgb:
            return self._image
        else:
            return self._image[:, :, ::-1]

    @property
    def width(self):
        return self.image().shape[1]

    @property
    def height(self):
        return self.image().shape[0]

    @property
    def size(self):
        return ImageSize(self.height, self.width)

    @staticmethod
    def coord2coord(x, y, raster_transform, coords_transform):
        return tuple([int(i) for i in (raster_transform * coords_transform(x, y))])
