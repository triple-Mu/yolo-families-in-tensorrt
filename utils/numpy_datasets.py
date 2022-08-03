from pathlib import Path
from random import shuffle

import cv2
import numpy as np

img_formats = [
    'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'
]


class NumpyhDataloader(object):

    def __init__(self,
                 root,
                 input_shape=(640, 640),
                 static_resize=False,
                 batch_size=1,
                 length=10000):
        self.root = []
        self.input_shape = input_shape
        self.static_resize = static_resize
        for path in Path(root).glob(f"*[{' '.join(img_formats)}]"):
            self.root.append(str(path))
        shuffle(self.root)
        length = min(length, len(self.root))
        self.batch_size = batch_size
        self.step = length // batch_size  # drop last
        self.root = self.root[:length]
        self.length = length
        self.input_shape = input_shape

    def get_batch(self):
        for _ in range(self.step):
            batch_images = []
            batch_ratio = []
            batch_dwdh = []
            roots = []
            for _ in range(self.batch_size):
                root = self.root.pop()
                bgr_im = cv2.imread(root)
                image, ratio, dwdh = self.letterbox(
                    bgr_im, self.input_shape, static_resize=self.static_resize)
                image = self.rgb2nchw(image)
                batch_images.append(image)
                batch_ratio.append(ratio)
                batch_dwdh.append(dwdh)
                roots.append(root)
            batch_images = np.concatenate(batch_images, 0)
            batch_images = np.ascontiguousarray(batch_images)
            batch_ratio = np.array(batch_ratio, dtype=np.float32)
            batch_dwdh = np.stack(batch_dwdh)
            yield batch_images, batch_ratio, batch_dwdh, root

    @staticmethod
    def letterbox(im,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  static_resize=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        if static_resize:
            im = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im, r, np.zeros((1, 4), dtype=np.float32)

            # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=color)  # add border
        dwdh = np.array([dw, dh, dw, dh]).astype(np.float32)
        return im, r, dwdh

    @staticmethod
    def rgb2nchw(im):
        im = np.transpose(im, [2, 0, 1])[::-1]
        im = np.expand_dims(im, 0)
        im = im.astype(np.float32)
        im = np.ascontiguousarray(im)
        return im


if __name__ == '__main__':
    loader = NumpyhDataloader(root='../calib_data', batch_size=12)

    iterloader = iter(loader.get_batch())

    for i in iterloader:
        print(i[0].shape)
