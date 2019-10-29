#!/usr/bin/env python2
"""Python script for aligning images taken with the MicaSense Altum multi-spectral camera.

Copyright (c) 2019. Sam Schofield. This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/sds53/altum_image_aligner/LICENSE.
No part of altum_image_aligner, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""

from __future__ import print_function

import cv2
import errno
import os

import numpy as np


def list_images_in_dir(directory):
    # TODO: don't search sub-directories
    files = []
    for r, d, f in os.walk(directory):
        for file_ in f:
            if file_.endswith(".tif"):
                files.append(os.path.join(r, file_))
    return files


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            print("Could not create directory")
            raise


def group_images(image_paths):
    """
    Create a dictionary that groups together the different spectral images at each time.
    Args:
        image_paths ([string]): list of all the images paths to group. Image paths should be of format IMG_n_m.tif where
        n is the sequence number and m is the spectral id.

    Returns:
        (dict): dictionary containing the group images. sequence_number (int): file_paths ([string]).
    """
    image_dict = {}
    for image_path in image_paths:
        image_name = image_path.split("/")[-1]
        sequence_number = int(image_name.split("_")[1])
        image_dict[sequence_number] = image_dict.get(sequence_number, []) + [image_path]
    return image_dict


def main(directory, output_directory):
    """

    """
    print("Finding images")
    image_paths = list_images_in_dir(directory)
    print("{} images found".format(len(image_paths)))

    print("Creating output directory")
    mkdir(output_directory)

    grouped_images = group_images(image_paths)

    for sequence_number, group_paths in grouped_images.items():
        print("Aligning sequence number: {}".format(sequence_number))
        group_paths = sorted(group_paths)
        shape = cv2.imread(group_paths[0]).shape
        bgr = np.empty((shape[0], shape[1], 3))
        for i, image_path in enumerate(group_paths[:3]):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            bgr[:, :, i] = image
        output_path = "{}/{}".format(output_directory, image_path.split("/")[-1])
        cv2.imwrite(output_path, bgr)
    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Aligns images from a MicaSense Altum camera')
    parser.add_argument('input_dir', help='Path to the directory containing the images')
    parser.add_argument('output_dir', help='Path to put the aligned images')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
