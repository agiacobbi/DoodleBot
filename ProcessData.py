# Alex Giacobbi
# 3/25/2020
#
# This program processes data
#
# Sources
# image processing code adapted from: https://github.com/martinohanlon/quickdraw_python/blob/master/quickdraw/data.py

import numpy as np 
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw


# constant values
PATH_NDJSON = "../airplane.ndjson"       # path to dataset directory
NUM_PER_CLASS = 10000                    # number of examples to extract


# Processes the 'drawing' component of the json object. Reads the each
# stroke, creating a list of coordinates. Uses PIL to create a BnW image
# of the strokes. Returns the image as a (256, 256) numpy array
#
# @param stroke_list a list of strokes in a drawing following form see
# https://github.com/googlecreativelab/quickdraw-dataset for more info
# [ 
#   [  // First stroke 
#     [x0, x1, x2, x3, ...],
#     [y0, y1, y2, y3, ...],
#     [t0, t1, t2, t3, ...]
#   ],
#   [  // Second stroke
#     [x0, x1, x2, x3, ...],
#     [y0, y1, y2, y3, ...],
#     [t0, t1, t2, t3, ...]
#   ],
#   ... // Additional strokes
# ]
# @return a (256, 256) numpy array representation of the doodle
def processStrokes(stroke_list):
    img = Image.new("L", (256, 256), 255)
    img_draw = ImageDraw.Draw(img)
    stroke_coordinates = []
    
    # read stroke_list structure into list of tuples (coordinates)
    for stroke in stroke_list:
        points = []
        for i in range(len(stroke[0])):
            points.append((stroke[0][i], stroke[1][i]))
        stroke_coordinates.append(points)

    # draw image using list of coordinates
    for stroke in stroke_coordinates:
        img_draw.line(stroke, fill=0, width=5)

    image_arr = np.array(img)
    return image_arr


# Processes an .ndjson file containing drawing data for a single
# class. Examples are read from .ndjson file and each stroke list is
# transformed into a (256, 256) numpy array. The processed array is 
# written to a file in the ../clean_data directory for retrieval
#
# @param path a path to the .ndjson file to be processed
# @param num_samples number of examples to extract from the .ndjson file
def processJsonClass(path, num_samples):
    images = []
    index = 0

    # get json objects
    with open(path) as f:
        examples_json = f.readlines()

    # iterate through json objects
    while len(images) < num_samples and index < len(examples_json):
        example = json.loads(examples_json[index])
    
        # only get drawings that were recognized
        if (example['recognized']):
            doodle_arr = processStrokes(example['drawing'])
            images.append(doodle_arr)

        index += 1

    # TODO: write data to a file


# process all classes in dir, save each class as np array
# with dimension (256, 256, NUM_SAMPLES)
if __name__ == "__main__":
    # loop through dataset and process each class
    # TODO: finish looping thorugh directory
    processJsonClass(PATH_NDJSON, NUM_PER_CLASS)