import logging
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

logger = logging.getLogger(__name__)


class CudaCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, cache_file):

        super().__init__()
        _, self.stream = cudart.cudaStreamCreate()
        self.shape = None
        self.dtype = np.dtype(np.float32)
        self.cache_file = Path(cache_file)
        self.loader = None
        self.batch = None
        self.batch_allocation = None
        self.idx = 0
        self.length = 0

    def set_image_batcher(self, dataloader):
        self.idx = 0
        self.length = dataloader.length
        self.shape = [dataloader.batch_size, 3, *dataloader.input_shape]
        self.loader = iter(dataloader.get_batch())
        _, self.batch_allocation = cudart.cudaMallocAsync(
            self.dtype.itemsize * np.prod(self.shape), self.stream)

    def get_batch_size(self):
        return self.shape[0] if self.shape is not None else 1

    def get_batch(self, names=['images']):
        try:
            batch = next(self.loader)
            img, ratio, dwdh, root = batch
            img /= 255
            cudart.cudaMemcpyAsync(
                self.batch_allocation, img.ctypes.data, img.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
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
