#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne
# License:

import os
import os.path
import shutil
import glob
import skimage.io
import matplotlib.pyplot as plt

from segmentation import segment_digit
from machine_learning import load_knowndata, load_unknowndata
from sklearn import svm

if __name__ == '__main__':
    output_dir = 'output'
    results_dir = 'results'
    for thisdir in (output_dir, results_dir):
        shutil.rmtree(thisdir, ignore_errors=True)
        os.makedirs(thisdir)
    show = False

    for filename in glob.glob('modern_digits/small*.png'):
        print(filename)
        # Load picture
        image = skimage.io.imread(filename, as_grey=True)
        # crop picture
        image = image[0:200, 47:]

        segment_digit(image, filename, output_dir, black_on_white=False, show=False)

    filenames = sorted(glob.glob('learnt_modern_digits/*-*.png'))
    training = load_knowndata(filenames)
    if show:
        plt.show()
        plt.close()

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=1e-8)

    # We learn the digits on the first half of the digits
    classifier.fit(training['data'], training['targets'])

    filenames = sorted(glob.glob(os.path.join(output_dir, '*.png')))
    unknown = load_unknowndata(filenames)

    filenames = set([os.path.splitext(os.path.basename(fn))[0].split('-')[0] for fn in filenames])

    for filename in filenames:
        fn = sorted(glob.glob(os.path.join(output_dir, filename + '*.png')))
        unknown = load_unknowndata(fn)
        # Now predict the value of the digit on the second half:
        # expected = digits.target[n_samples / 2:]
        predicted = classifier.predict(unknown['data'])

        result = ''
        for pred, image, name in zip(predicted, unknown['images'], unknown['name']):
            result += str(pred)

        # Check
        fn = os.path.join('modern_digits', filename  + '.png')
        image = skimage.io.imread(fn)
        plt.imshow(image[:150, 56:], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Predicted: %s' % result)
        plt.savefig(os.path.join(results_dir, filename))
