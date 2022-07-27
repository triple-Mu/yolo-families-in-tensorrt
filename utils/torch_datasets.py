from pathlib import Path
from random import shuffle

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

img_formats = [
    'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'
]

TORCH = True


class TorchDataset(Dataset):

    def __init__(self,
                 root,
                 input_shape=(640, 640),
                 static_resize=False,
                 length=10000):
        super(TorchDataset, self).__init__()

        self.root = []
        self.input_shape = input_shape
        self.static_resize = static_resize
        for path in Path(root).glob(f"*[{'.'.join(img_formats)}]"):
            self.root.append(str(path))
        shuffle(self.root)
        length = min(length, len(self.root))
        self.root = self.root[:length]
        self.length = length
        self.input_shape = input_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        root = self.root[index]
        bgr_im = cv2.imread(root)
        image, ratio, dwdh = self.letterbox(bgr_im,
                                            self.input_shape,
                                            static_resize=self.static_resize)
        image = self.rgb2nchw(image)
        image = image.astype(np.float32)
        if TORCH:
            image = torch.from_numpy(image)
            dwdh = torch.from_numpy(dwdh)
        return image, ratio, dwdh, root

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
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        return im

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, ratio, dwdh, root = zip(*batch)
        if TORCH:
            img = torch.cat(img, 0)
            ratio = torch.tensor(ratio, dtype=torch.float32)
            dwdh = torch.stack(dwdh)
        else:
            img = np.concatenate(img, 0)
            ratio = np.array(ratio, dtype=np.float32)
            dwdh = np.stack(dwdh)
        return img, ratio, dwdh, root


if __name__ == '__main__':
    data = TorchDataset(root='../calib_data')

    loader = DataLoader(data,
                        batch_size=10,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=data.collate_fn)
    iterloader = iter(loader)

    print(next(iterloader)[0].shape)
    print(next(iterloader)[3])
    print(next(iterloader)[0].shape)
    print(next(iterloader)[3])
