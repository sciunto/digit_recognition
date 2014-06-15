#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: Simplified BSD

import os

import numpy as np
import pylab as pl

import skimage.io

def load_knowndata(filenames):
    training = {'images': [], 'targets': [], 'data': [], 'name': []}

    for index, filename in enumerate(filenames):
        target = os.path.splitext(os.path.basename(filename))[0]
        target = int(target.split('-')[0])
        image = skimage.io.imread(filename)
        training['targets'].append(target)
        training['images'].append(image)
        training['name'].append(filename)
        training['data'].append(image.flatten().tolist())
        pl.subplot(6, 5, index + 1)
        pl.axis('off')
        pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
        pl.title('Training: %i' % target)

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    training['images'] = np.array(training['images'])
    training['targets'] = np.array(training['targets'])
    training['data'] = np.array(training['data'])
    return training

def load_unknowndata(filenames):
    training = {'images': [], 'targets': [], 'data': [], 'name': []}

    for index, filename in enumerate(filenames):
        image = skimage.io.imread(filename)
        training['targets'].append(-1)  # Target = -1: unkown
        training['images'].append(image)
        training['name'].append(filename)
        training['data'].append(image.flatten().tolist())

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    training['images'] = np.array(training['images'])
    training['targets'] = np.array(training['targets'])
    training['data'] = np.array(training['data'])
    return training
