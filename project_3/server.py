"""
ECE196 Face Recognition Project
Author: W Chen

What this script should do:
0. assume the local host is already connected to ec2 instance
1. load model with saved weights
2. use a loop to do:
    2.1. check if a new face image is saved in IMG_SRC_DIR.
    2.2. process and classify the image
    2.3. save the classification result in RESULT_DIR.
"""


import glob, os, time, cv2
import numpy as np
from keras.models import load_model

# TODO: some parameters you might use
IMG_SRC_DIR = ''  # ec2
RESULT_DIR = ''  # ec2
RESULT_FILE_NAME = 'result.txt'


def check_new_file(path):
    """
    check if a new file is available in the given directory
    :param path: path of the directory to check
    :return: path of the file if available; None if not available
    """
    new_file_path = None
    file_path = os.path.join(path,'*')
    file_list = sorted(glob.glob(file_path))
    if len(file_list) == 1:
        new_file_path = file_list[0]
    return new_file_path


def classify(file_path, model):
    """
    classify a face image
    :param file_path: path of face image
    :param model: model to use
    :return: classification results label and confidence
    """
    img_height, img_width, num_channel = 224, 224, 3
    mean_pixel = np.array([104., 117., 123.]).reshape((1, 1, 3))

    # TODO: use opencv to read and resize image to standard dimensions
    # TODO: subtract mean_pixel from that image, name the final image as new_img

    x = np.expand_dims(new_img, axis=0)

    # TODO: use network to predict x, get label and confidence of prediction
    # TODO: label is a number, which correspond to the same number you give to the folder when you organized data

    return label, conf


def write_result(label, conf):
    """
    write label and confidence to a txt file
    :param label: predicted class
    :param conf: confidence of prediction
    """
    # open file to write in
    result_file = open(os.path.join(RESULT_DIR, RESULT_FILE_NAME), 'w')
    # TODO: convert the label to a name. Eg. if the label of your face is 20, save your name as "name"

    result = ','.join([name, str(conf)])
    result_file.write(result)
    result_file.close()
    return


def main():
    # TODO: read saved weights and name it model

    model.summary()

    print 'Starting ...'

    # TODO: use a loop to check for file
    while True:
        # TODO: check if a new face image is saved

        # TODO: if new face image available, classify it and save result to a txt file
        # TODO: wait for 10 seconds (by then, the result should've been fetched by RPi) and delete image and result files

        time.sleep(1)


if __name__ == "__main__":
    main()
