from gluoncv.data.segbase import SegmentationDataset
from gluoncv.data.base import VisionDataset
from PIL import Image, ImageOps, ImageFilter
import platform, os
import logging
import mxnet as mx
import random
from mxnet import cpu, nd
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class IceContrast(SegmentationDataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, base_dir='DENTIST', root=os.path.join('~', 'Nutstore Files', 'Dataset'),
                 split='train', mode=None, transform=None, **kwargs):
        super(IceContrast, self).__init__(root, split, mode, transform, **kwargs)

        self.base_dir = base_dir
        self._root = os.path.expanduser(os.path.join(root, base_dir))
        self._transform = transform
        self._split = split
        self.mode = mode
        self._items = self._load_items(split)
        if base_dir == 'DENTIST':
            self._anno_path = os.path.join('{}', 'masks/', '{}_pixels0.png')
            self._image_path = os.path.join('{}', 'images', '{}.png')
        elif base_dir == 'Iceberg':
            self._anno_path = os.path.join('{}', 'labels/mask/', '{}_pixels0.png')
            self._image_path = os.path.join('{}', 'images', '{}.png')
        else:
            raise ValueError("Unknown base dir")

    def _load_items(self, split):
        """Load individual image indices from splits."""
        ids = []
        root = self._root
        lf = os.path.join(root, split + '.txt')
        with open(lf, 'r') as f:
            ids += [(root, line.strip()) for line in f.readlines()]

        random.shuffle(ids)
        return ids

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label_path = self._anno_path.format(*img_id)

        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'test':
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, img_id[-1]
        mask = Image.open(label_path)
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            if self.base_dir == 'DENTIST':
                img, mask = self._testval_sync_transform(img, mask)
            else:
                img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = nd.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, mask#, img_id[-1]

    def __len__(self):
        return len(self._items)

    @property
    def classes(self):
        """Category names."""
        return ('iceberg')

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
