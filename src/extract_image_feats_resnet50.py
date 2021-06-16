#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import os, sys

import numpy as np

from PIL import Image

import torch
from torch.autograd import Variable
import torch.utils.data as data

from torchvision.models import resnet50
from torchvision import transforms

# Note: from https://github.com/multi30k/dataset/blob/master/scripts/feature-extractor
# This script uses the PyTorch's pre-trained ResNet-50 CNN to extract
#   res4f_relu convolutional features of size 1024x14x14
#   avgpool features of size 2048D
# We reproduced ImageNet val set Top1/Top5 accuracy of 76.1/92.8 %
# as reported in the following web page before extracting the features:
#   http://pytorch.org/docs/master/torchvision/models.html
#
# We save the final files as 16-bit floating point tensors to reduce
# the size by 2x. We confirmed that this does not affect the above accuracy.
#
# Organization of the image folder:
#  In order to extract features from an arbitrary set of images,
#  you need to create a folder with a file called `index.txt` in it that
#  lists the filenames of the raw images in an ordered way.
#    -f /path/to/images/train  --> train folder contains 29K images
#                                  and an index.txt with 29K lines.
#


class ImageFolderDataset(data.Dataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images not attached to any other
    label.

    Arguments:
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.
    """
    def __init__(self, image_folder, file_names, resize=None, crop=None):
        self.root = image_folder
        self.index = file_names

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        if not self.index.exists():
            raise(RuntimeError(
                "{} does not exist.".format(self.index)))

        self.image_files = []
        with self.index.open() as f:
            for fname in f:
                fname = self.root / fname.strip()
                assert fname.exists(), "{} does not exist.".format(fname)
                self.image_files.append(str(fname))

    def read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        return self.read_image(self.image_files[idx])

    def __len__(self):
        return len(self.image_files)


def resnet_forward(cnn, x):
    x = cnn.conv1(x)
    x = cnn.bn1(x)
    x = cnn.relu(x)
    x = cnn.maxpool(x)

    x = cnn.layer1(x)
    x = cnn.layer2(x)
    res4f_relu = cnn.layer3(x)
    res5e_relu = cnn.layer4(res4f_relu)

    avgp = cnn.avgpool(res5e_relu)
    avgp = torch.flatten(avgp, 1)
    return res4f_relu, res5e_relu, avgp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract-cnn-features')
    parser.add_argument('-i', '--image-folder', type=str, required=True,
                        help='Folder to image files i.e. /images/train')
    parser.add_argument("-f", "--file_names", type=str, required=True,
                        help="""File containing a list with image file names.""")
    
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='Batch size for forward pass.')
    parser.add_argument('-m', '--model-file', type=str, required=True,
                        help='File containing resnet50 weights (download from https://download.pytorch.org/models/resnet50-0676ba61.pth).')
    
    parser.add_argument('-o', '--output-prefix', type=str, required=True,
                        help='Output file prefix. Ex: out/train/resnet50')

    # Parse arguments
    args = parser.parse_args()

    image_folder = Path(args.image_folder).expanduser().resolve()
    file_names = Path(args.file_names).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    model_file = Path(args.model_file).expanduser().resolve()

    print("Output folder: %s" % str(output_prefix.parent))
    os.makedirs(output_prefix.parent, exist_ok=True)

    if not model_file.exists():
        raise(RuntimeError("%s does not exist." % str(model_file)))

    # Create dataset
    dataset = ImageFolderDataset(image_folder, file_names,
                                 resize=256, crop=224)
    print('Image folder: %s' % args.image_folder)

    loader = data.DataLoader(dataset, batch_size=args.batch_size)

    print('Creating CNN instance.')
    cnn = resnet50(pretrained=False)
    #resnet_dict = torch.hub.load_state_dict_from_url(
    #    "", file_name="blah/"+str(model_file), progress=True)
    resnet_dict = torch.load(str(model_file))
    cnn.load_state_dict(resnet_dict)

    # Remove final classifier layer
    del cnn.fc

    # Move to GPU and switch to evaluation mode
    cnn.cuda()
    cnn.train(False)

    # Create memmaped files
    res4f_feats = np.lib.format.open_memmap(
        str(output_prefix) + "-resnet50-res4frelu.npy", mode="w+",
        dtype=np.float16, shape=(len(dataset), 1024, 14, 14))
    res5e_feats = np.lib.format.open_memmap(
        str(output_prefix) + "-resnet50-res5erelu.npy", mode="w+",
        dtype=np.float16, shape=(len(dataset), 2048, 7, 7))
    pool_feats = np.lib.format.open_memmap(
        str(output_prefix) + "-resnet50-avgpool.npy", mode="w+",
        dtype=np.float16, shape=(len(dataset), 2048))

    n_batches = int(np.ceil(len(dataset) / args.batch_size))

    bs = args.batch_size

    for bidx, batch in enumerate(loader):
        x = batch.cuda()
        with torch.no_grad():
            res4f, res5e, avgpool = resnet_forward(cnn, x)

        pool_feats[bidx * bs: (bidx + 1) * bs] = avgpool.data.cpu().numpy().astype(np.float16)
        res4f_feats[bidx * bs: (bidx + 1) * bs] = res4f.data.cpu().numpy().astype(np.float16)
        res5e_feats[bidx * bs: (bidx + 1) * bs] = res5e.data.cpu().numpy().astype(np.float16)

        print('{:3}/{:3} batches completed.'.format(
            bidx + 1, n_batches), end='\r')
        sys.stdout.flush()

    # Save the files
    res4f_feats.flush()
    res5e_feats.flush()
    pool_feats.flush()
 
