"""This program processes the simplified .ndjson QuickDraw dataset

Visits each .ndjson file in directory containing drawing data defined here as 
'../dataset/'. For each class of drawing, a number of examples are extracted and
processed into 256 X 256 numpy arrays. Processed data is written to a .dat file
in a separate directory.

Authors:
    Alex Giacobbi

Date:
    3/25/2020

Sources:
    image processing code adapted from: https://github.com/martinohanlon/quickdraw_python/blob/master/quickdraw/data.py
"""

import numpy as np 
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw


PATH_DATASET_DIR = "./dataset/"      # path to dataset directory
PATH_CLEAN_DATA = "./processed_data/vehicle_data/"
MAX_SAMPLES = 50000       # number of examples to extract
# CLASSES = ['ant', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 
#            'crab', 'crocodile', 'dog', 'dolphin', 'dragon', 'duck', 'elephant',
#            'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'monkey',
#            'mosquito', 'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin',
#            'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion', 'sea turtle', 
#            'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 
#            'tiger', 'whale', 'zebra']
CLASSES = ['ambulance', 'bus', 'car', 'firetruck', 'pickup truck', 'police car',
           'school bus', 'tractor', 'truck']


def process_strokes(stroke_list):
    """Processes the a stroke representation for a drawing.

    Processes drawing represented as list of strokes. Transforms list of strokes
    into a BnW image then to a numpy array

    Args:
        stroke_list: a drawing represented as a list of sequential strokes with
            this structure:

            [
            [  // First stroke 
                [x0, x1, x2, x3, ...],
                [y0, y1, y2, y3, ...]
            ],
            [  // Second stroke
                [x0, x1, x2, x3, ...],
                [y0, y1, y2, y3, ...]
            ],
            ... // Additional strokes
            ]
            adapted from https://github.com/googlecreativelab/quickdraw-dataset
            (the timing data is omitted for the simplified dataset)

    Returns:
        A numpy array with dimension 256 x 256 where each cell represents a 
        pixel in the doodle. 
    """

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

    img = img.resize((28, 28), resample=Image.LANCZOS)
    image_arr = np.array(img)
    image_arr = (image_arr - 127.5) / 127.5
    return image_arr


def process_json_class(path, class_name, num_samples):
    """Extracts a number of examples from an .ndjson file

    Processes an .ndjson file containing drawing data for a single class. 
    Examples are read from .ndjson file and each stroke list is transformed into
    a 256 x 256 numpy array. The processed arrays are compiled into a single
    matrix which is written to a file located in ../clean_data for retrieval

    Args:
        path: a path to the .ndjson file to be processed
        class_name: name of class being processed
        num_samples: the number of samples to extract from the .ndjson file. If
            the number of samples in the files is less than num_samples, will
            finish extraction once all samples are processed
    """

    images = []
    index = 0
    class_name = os.path.splitext(class_name)[0]

    print('Processing drawings of ' +  class_name + '\n')

    # get json objects
    with open(path) as f:
        examples_json = f.readlines()

    # iterate through json objects
    while len(images) < num_samples and index < len(examples_json):
        example = json.loads(examples_json[index])
    
        # only get drawings that were recognized
        if (example['recognized']):
            doodle_arr = process_strokes(example['drawing'])
            images.append(doodle_arr)

        index += 1

    processed_drawings = np.array(images)
    np.save(PATH_CLEAN_DATA + class_name, processed_drawings)

    print("done.")

def main():
    if not os.path.exists(PATH_CLEAN_DATA):
        os.makedirs(PATH_CLEAN_DATA)
    # loop through dataset and process each class
    # TODO: finish looping thorugh directory
    for filename in os.listdir(PATH_DATASET_DIR):
        class_name = os.path.splitext(filename)[0]
        if filename.endswith('.ndjson') and class_name in CLASSES:
            process_json_class(PATH_DATASET_DIR + filename, filename, 
                               MAX_SAMPLES)


if __name__ == "__main__":
    main()