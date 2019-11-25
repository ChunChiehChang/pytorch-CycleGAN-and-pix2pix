#!/usr/bin/env python3

import argparse
import os
import numpy as np
import random
import skimage
import math
from skimage import io, transform

parser = argparse.ArgumentParser(description="Converts Kaldi file format")
parser.add_argument('database_path', type=str, help='Path to data')
parser.add_argument('out_dir', type=str, help='directory to output files')
parser.add_argument('--head', type=int, default=-1, help='limit on number of data')
parser.add_argument('--feat-dim', type=int, default=60, help='y image height')
parser.add_argument('--feat-length', type=int, default=750, help='x image length')
parser.add_argument('--num-channels', type=int, default=3, help='number of color channels')
args = parser.parse_args()

def get_scaled_image(im):
    scale_size = args.feat_dim
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    nx = int(scale_size)
    ny = int(scale * sx)
    im = transform.resize(im, (nx, ny))
    return im

def pad_length(im):
    im_length = im.shape[1]
    im_height = im.shape[0]
    #left_padding = math.floor((args.feat_length-im_length)/2.0)
    #right_padding = args.feat_length - im_length - left_padding
    #im = np.concatenate((255 * np.ones((im_height, left_padding, args.num_channels), dtype=int), im), axis=1)
    #im = np.concatenate((im, 255 * np.ones((im_height, right_padding, args.num_channels), dtype=int)), axis=1)
    right_padding = args.feat_length-im_length
    im = np.concatenate((im, 255 * np.ones((im_height, right_padding, args.num_channels), dtype=int)), axis=1)
    return im

### main ###
text_file1 = os.path.join(args.out_dir, 'text_train')
text_fh1 = open(text_file1, 'w', encoding='utf-8')
text_file2 = os.path.join(args.out_dir, 'text_test')
text_fh2 = open(text_file2, 'w', encoding='utf-8')
count = 0
train_count = 0
test_count = 0
transcription_file = os.path.join(args.database_path, 'text.old')
transcription_dict = {}
images_scp = os.path.join(args.database_path, 'images.scp')
with open(transcription_file) as f:
    for line in f:
        line_vect = line.strip().split()
        transcription = ''.join(line_vect[1:])
        transcription_dict[line_vect[0]] = transcription

with open(images_scp) as f:
    for line in f:
        line_vect = line.strip().split()
        image_id = line_vect[0]
        image_path = line_vect[1]
        image_transcription = transcription_dict[image_id]
        im = io.imread(image_path)
        im = get_scaled_image(im)
        if im.shape[1] < args.feat_length - 2 and len(list(image_transcription)) > 3 and (count < args.head or args.head < 0):
            count += 1
            im = pad_length(skimage.img_as_ubyte(im))
            if random.randint(0,20) == 0:
                test_count += 1
                image_filepath_scaled = os.path.join(args.out_dir, 'images_test', image_id + '.png')
                io.imsave(image_filepath_scaled, im)
                text_fh2.write(str(test_count) + ' ' + image_id + ' ' + image_filepath_scaled + ' ' + image_transcription + '\n')
            else:
                train_count += 1
                image_filepath_scaled = os.path.join(args.out_dir, 'images_train', image_id + '.png')
                io.imsave(image_filepath_scaled, im)
                text_fh1.write(str(train_count) + ' ' + image_id + ' ' + image_filepath_scaled + ' ' + image_transcription + '\n')
