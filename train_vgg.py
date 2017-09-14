"""
ECE196 Face Recognition Project
Author: W Chen

Use this as a template to:
1. load weights for vgg16
2. load images
3. finetune network
4. save weights
"""


from keras.models import Model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import glob
import os
import cv2
import random

img_height, img_width, num_channel = 224, 224, 3
mean_pixel = np.array([104., 117., 123.]).reshape((1,1,3))
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
num_epochs = 10
batch_size = 16
num_classes = 20
task_name = 'fr_withNewFrontal_3'


def load_model():
    # build the VGG16 network
    base_model = applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                    input_shape=(img_height, img_width, num_channel))
    print('Model weights loaded.')
    x = base_model.output
    flat = Flatten()(x)
    hidden = Dense(256, activation='relu')(flat)
    drop = Dropout(0.5)(hidden)
    predictions = Dense(num_classes, activation='softmax')(drop)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Model building complete.'

    # first: train only the top layers (which were randomly initialized)
    for layer in model.layers[:15]:
        layer.trainable = False
    print 'Freezing conv layers complete.'
    model.summary()

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print 'Compile model...'
    return model


def load_data(src_path):
    # under train/val/test dirs, each class is a folder with numerical numbers
    class_path_list = sorted(glob.glob(os.path.join(src_path, '*')))
    image_path_list = []
    for class_path in class_path_list:
        image_path_list += sorted(glob.glob(os.path.join(class_path, '*jpg')))
    random.shuffle(image_path_list)
    image_counter = len(image_path_list)
    print 'This set has {} images.'.format(image_counter)
    X = np.zeros((image_counter, img_height, img_width, num_channel))
    Y = np.zeros((image_counter, 1))
    # read images and labels
    for i in range(image_counter):
        image_path = image_path_list[i]
        label = int(image_path.split('/')[-2])
        image = cv2.imread(image_path, 1)
        image = process_image(image)
        #image = cv2.resize(image, (img_height, img_width)) - mean_pixel
        #image = image.reshape((img_height, img_width, num_channel))
        image -= mean_pixel
        X[i, :, :, :] = image
        Y[i, :] = label
    Y = to_categorical(Y, num_classes)
    return X, Y


def main():
    model = load_model()
    return


if __name__ == '__main__':
    main()
