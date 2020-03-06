from itertools import product
from typing import Union, Tuple, Sequence, Iterator

from cadaster.gis import Raster
import numpy as np
import torch
from tqdm.autonotebook import tqdm

class Predictor:

    def __init__(self, model, tile_size: Union[int, Tuple[int, int]]=(500, 500), margin: int = 0, min_overlap: float = 0.3):
        self.model = model
        if isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)

        self.margin = margin
        self.min_overlap = min_overlap

    def predict(self, raster: Raster, batch_size: int = 4, device: str = 'cpu') -> np.array:
        h, w = raster.size
        y_step = self.compute_step(h, self.tile_size[0], self.margin, self.min_overlap)
        x_step = self.compute_step(w, self.tile_size[1], self.margin, self.min_overlap)
        y_pos = np.round(np.arange(y_step + 1) / y_step * (h - self.tile_size[0])).astype(np.int32)
        x_pos = np.round(np.arange(x_step + 1) / x_step * (w - self.tile_size[1])).astype(np.int32)

        image = raster.image()
        margin = self.margin
        image_padded = np.pad(image, ((margin,margin),(margin,margin),(0,0)), mode='reflect')

        counts = np.zeros(raster.size, dtype=np.int)
        probs_sum = np.zeros([7] + list(raster.size))

        for pos in tqdm(batch_items(list(product(y_pos, x_pos)), batch_size),
                        total=(len(y_pos)*len(x_pos))//batch_size+1):
            crops = np.stack([pos2crop(image_padded, x, y, self.tile_size, self.margin) for x, y in pos])
            probs = imgs2preds(crops, self.model, device, self.margin)
            for idx, (y, x) in enumerate(pos):
                counts[y:y+self.tile_size[0], x:x+self.tile_size[1]] += 1
                probs_sum[:,y:y+self.tile_size[0], x:x+self.tile_size[1]] += probs[idx]
        probs = probs_sum/counts

        return probs

    @staticmethod
    def compute_step(size: int, tile_size: int, margin:int, min_overlap: float) -> int:
        return np.ceil((size - tile_size - margin) / ((tile_size - margin) * (1 - min_overlap)))


def batch_items(iterable: Sequence, batch_size: int = 1) -> Iterator:
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def imgs2torch(imgs, device: str = 'cpu') -> torch.Tensor:
    return torch.from_numpy(imgs.transpose(0, 3, 1, 2) / 255).float().to(device)


def imgs2preds(imgs, model, device='cpu', margin=16) -> torch.Tensor:
    imgs_torch = imgs2torch(imgs, device)
    with torch.no_grad():
        preds = torch.sigmoid(model.to(device)(imgs_torch)).cpu().numpy()[:, :, margin:-margin, margin:-margin]
    return preds


def pos2crop(image: np.array, x: int, y: int,
             tile_size: Union[int, Tuple[int, int]]=(500, 500), margin: int = 0) -> np.array:
    return image[y:y+tile_size[0]+margin*2,x:x+tile_size[1]+margin*2]