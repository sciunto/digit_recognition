#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: Simplified BSD

import glob
import os

import numpy as np
import pylab as pl

from sklearn import svm, metrics
import skimage.io

def load_knowndata(filenames):
    training = {'images': [], 'targets': [], 'data' : []}

    for index, filename in enumerate(filenames):
        target = int(os.path.splitext(os.path.basename(filename))[0])
        image = skimage.io.imread(filename)
        training['targets'].append(target)
        training['images'].append(image)
        training['data'].append(image.flatten().tolist())
        pl.subplot(2, 5, index + 1)
        pl.axis('off')
        pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
        pl.title('Training: %i' % target)

    ## To apply an classifier on this data, we need to flatten the image, to
    ## turn the data in a (samples, feature) matrix:
    n_samples = len(training['images'])
    training['images'] = np.array(training['images'])
    training['targets'] = np.array(training['targets'])
    training['data'] = np.array(training['data'])
    return training

def load_unknowndata(filenames):
    training = {'images': [], 'targets': [], 'data' : []}

    for index, filename in enumerate(filenames):
        image = skimage.io.imread(filename)
        training['targets'].append(-1) # Target = -1: unkown
        training['images'].append(image)
        training['data'].append(image.flatten().tolist())

    ## To apply an classifier on this data, we need to flatten the image, to
    ## turn the data in a (samples, feature) matrix:
    n_samples = len(training['images'])
    training['images'] = np.array(training['images'])
    training['targets'] = np.array(training['targets'])
    training['data'] = np.array(training['data'])
    return training

if __name__ == '__main__':
    filenames = sorted(glob.glob('learn/*.png'))
    training = load_knowndata(filenames)
    pl.show()
    pl.close()

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(training['data'], training['targets'])

    filenames = sorted(glob.glob('output/*.png'))
    unknown = load_unknowndata(filenames)

    # Now predict the value of the digit on the second half:
    #expected = digits.target[n_samples / 2:]
    predicted = classifier.predict(unknown['data'])

    for pred, image in zip(predicted, unknown['images']):
        pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
        pl.title('Predicted: %i' % pred)
        pl.show()
