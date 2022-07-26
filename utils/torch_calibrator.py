import logging
from pathlib import Path

import tensorrt as trt
import torch

logger = logging.getLogger(__name__)


class TorchCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, cache_file, device=None):

        super().__init__()
        self.device = torch.device(
            f'cuda:{device}') if device is None else device
        self.shape = None
        self.dtype = torch.float32
        self.cache_file = Path(cache_file)
        self.loader = None
        self.batch = None
        self.batch_allocation = None
        self.idx = 0
        self.length = 0

    def set_image_batcher(self, dataloader):
        self.idx = 0
        self.length = dataloader.dataset.length
        self.shape = [
            dataloader.batch_size, 3, *dataloader.dataset.input_shape
        ]
        self.loader = iter(dataloader)
        self.batch = torch.empty(self.shape, dtype=self.dtype).to(self.device)
        self.batch_allocation = self.batch.data_ptr()

    def get_batch_size(self):
        return self.shape[0] if self.shape is not None else 1

    def get_batch(self, names=['images']):
        try:
            batch = next(self.loader)
            img, ratio, dwdh, root = batch
            img = img.to(self.device)
            img /= 255
            self.batch.data[...] = img.data[...]
            logger.info(f'Calibrating batch {self.idx} / {self.length}')
            self.idx += 1
            return [int(self.batch_allocation)]
        except StopIteration:
            logger.info('Finished calibration batches')
            return None

    def read_calibration_cache(self):
        if self.cache_file.exists():
            logger.info(f'Using calibration cache file: {self.cache_file}')
            return self.cache_file.read_bytes()

    def write_calibration_cache(self, cache):
        logger.info(f'Writing calibration cache data to: {self.cache_file}')
        self.cache_file.write_bytes(cache)
