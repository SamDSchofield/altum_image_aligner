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


def align_image(reference_image, image):
    """
    Aligns image2 to image1 using cv2's findTransformECC and warpAffine.
    Taken from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    Args:
        reference_image (np.array): The image to align to.
        image (np.array): The image to align.

    Returns:
        (np.array): image aligned to the reference_image
    """
    #
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find size of reference_image
    sz = reference_image.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    # termination_eps = 1e-10
    termination_eps = 1e-5

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(image, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned


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
    Take the images from the given directory, align the images taken for each sequence number and save them to the
    output directory.

    The images are aligned using cv2.findTransformECC and cv2.warpAffine.

    Args:
        directory: Path to the directory containing the un-aligned image
        output_directory: Path to the directory to save the aligned images.
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
        reference_image = cv2.imread(group_paths[0])
        output_path = "{}/{}".format(output_directory, group_paths[0].split("/")[-1])
        cv2.imwrite(output_path, reference_image)
        for image_path in group_paths[1:]:
            if not image_path.endswith("6.tif"):  # Skip the thermal images
                print(image_path)
                image = cv2.imread(image_path)
                aligned_image = align_image(reference_image, image)
                output_path = "{}/{}".format(output_directory, image_path.split("/")[-1])
                print(output_path)
                print()
                cv2.imwrite(output_path, aligned_image)
    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Aligns images from a MicaSense Altum camera')
    parser.add_argument('input_dir', help='Path to the directory containing the images')
    parser.add_argument('output_dir', help='Path to put the aligned images')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
