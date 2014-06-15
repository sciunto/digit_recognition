#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne
# License:

import os.path
import glob

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage.io
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import regionprops, label
from skimage.color import label2rgb


def segment_digit(image, filename, output_dir, digit_height=57, digit_width=33,
                  border=7, black_on_white=True, closingpx=8):
    """
    Segement each digit of a picture

    :param image: grey scale picture
    :param filename: filename of the picture source
    :param output_dir: path for the output
    :param digit_height: height of a digit
    :param digit_width: width of a digit
    :param border: pixels to shift the border
    :param black_on_white: black digit on clear background
    :param closingpx: number of pixels to close
    """
    # apply threshold
    thresh = threshold_otsu(image)
    if black_on_white:
        bw = closing(image > thresh, square(closingpx))
    else:
        bw = closing(image < thresh, square(closingpx))

    filled = ndimage.binary_fill_holes(bw)

    # remove artifacts connected to image border
    cleared = filled.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(filled, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)

    regions = regionprops(label_image)

    for item, region in enumerate(regions):
        # skip small elements
        if region['Area'] < 100:
            continue

        # draw rectangle around segmented digits
        minr, minc, maxr, maxc = region['BoundingBox']
        minr = maxr - digit_height
        minc = maxc - digit_width
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        newname = os.path.splitext(os.path.basename(filename))[0] + '-' + str(item) + '.png'
        skimage.io.imsave(os.path.join(output_dir, newname), image[minr:maxr+border, minc:maxc+border])
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    for filename in glob.glob('data/im/*9.png'):
        # Load picture
        image = skimage.io.imread(filename, as_grey=True)
        # crop picture
        image = image[350:, :]

        output_dir = 'output'
        os.makedirs(output_dir)
        segment_digit(image, filename, output_dir, black_on_white=True)

