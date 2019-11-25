"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import re
import unicodedata

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_ctc(dir, lang, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    phones = {}
    count = 0
    with open(os.path.join(lang, 'local', 'dict', 'phones.txt')) as f:
        for line in f:
            line = unicodedata.normalize('NFKC', line)
            phones[line.strip()] = count
            count = count + 1
    phones['NULL'] = count

    pattern = '|'.join(sorted(re.escape(k) for k in phones))
    with open(os.path.join(dir, 'text')) as f:
        for line in f:
            line = unicodedata.normalize('NFKC', line)
            line_vect = line.split(' ')
            path = os.path.join(dir, 'images', line_vect[0] + '.png')
            text = ' '.join(line_vect[1:])
            text = re.sub(pattern, lambda x: str(phones.get(x.group(0))), text)
            text = text.replace(' ', '')
            text_vect = [int(x) if x.isnumeric() else phones['NULL'] for x in list(text)]
            images.append([text_vect, path])
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
