import glob
import tensorflow as tf
import random
import numpy as np
import utils
from deep_calib import DeepCalib
import time
import os

# training parameters
batch_size = 50
n_epochs = 1000

IMAGE_FILE_PATH_DISTORTED = "app/DeepCalib/dataset_discrete/"

focal_start = 40
focal_end = 500
dist_end = 1.2
classes_focal = list(np.arange(focal_start, focal_end+1, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)


def get_paths(IMAGE_FILE_PATH_DISTORTED):
    paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.jpg")
    paths_train.sort()
    parameters = []
    labels_focal_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_train.append((curr_parameter - focal_start*1.) / (focal_end*1. - focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_train.append(curr_parameter/1.2)

    c = list(zip(paths_train, labels_focal_train,labels_distortion_train))
    random.shuffle(c)
    paths_train, labels_focal_train,labels_distortion_train = zip(*c)
    paths_train, labels_focal_train, labels_distortion_train = list(paths_train), list(labels_focal_train), list(labels_distortion_train)
    labels_train = [list(a) for a in zip(labels_focal_train, labels_distortion_train)]

    paths_valid = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'valid/' + "*.jpg")
    paths_valid.sort()
    parameters = []
    labels_focal_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_valid.append((curr_parameter-focal_start*1.)/(focal_end*1.+1.-focal_start*1.)) #normalize bewteen 0 and 1
    labels_distortion_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_valid.append(curr_parameter/1.2)

    c = list(zip(paths_valid, labels_focal_valid, labels_distortion_valid))
    random.shuffle(c)
    paths_valid, labels_focal_valid, labels_distortion_valid = zip(*c)
    paths_valid, labels_focal_valid, labels_distortion_valid = list(paths_valid), list(labels_focal_valid), list(labels_distortion_valid)
    labels_valid = [list(a) for a in zip(labels_focal_valid, labels_distortion_valid)]

    return paths_train, labels_train, paths_valid, labels_valid


if __name__ == '__main__':

    train_paths, train_labels, valid_paths, valid_labels = get_paths(IMAGE_FILE_PATH_DISTORTED)

    train_images = [utils.preprocess_image(x) for x in train_paths]
    validation_images = [utils.preprocess_image(x) for x in valid_paths]

    training_steps = len(train_images) // batch_size
    validation_steps = len(validation_images) // batch_size

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).repeat(training_steps * n_epochs)
    validation = tf.data.Dataset.from_tensor_slices((validation_images, valid_labels)).batch(batch_size).repeat(validation_steps * n_epochs)

    deep_calib = DeepCalib()
    model = deep_calib()

    training_start = time.time()

    model.fit(
        train,
        batch_size=batch_size,
        steps_per_epoch=(len(train_images) // batch_size),
        epochs=n_epochs,
        shuffle=True,
        validation_data=validation,
        validation_steps=(len(validation_images) // batch_size),
        use_multiprocessing=True,
        workers=os.cpu_count() * 2
    )

    training_end = time.time()

    model.save('app/DeepCalib/DeepCalibModel')

    print('Trained Model in ' +  utils.print_run_time(training_end - training_start))



