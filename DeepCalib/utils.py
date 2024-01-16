import tensorflow as tf
import math
import cv2
import random
import numpy as np
import keras.backend as K
from keras.layers import Rescaling
from PIL import Image, ImageOps
from keras.applications.imagenet_utils import preprocess_input
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
PI = math.pi
#Create Dictionaries
ROTATE = {}
ANGLE = {}
SLICE = {}


def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.

    """

    return 180 - abs(abs(x-y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the
    true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def slice_image(image, slice_dict):

    if slice_dict == 0:
        sliced_image = image

    else:
        map_coordinate = SLICE[slice_dict]
        map_x = map_coordinate[0]
        map_y = map_coordinate[1]
        # np.save(PROJECT_FILE_PATH+"x_generated_sliced.npy", map_x)
        # np.save(PROJECT_FILE_PATH+"y_generated_sliced.npy", map_y)
        sliced_image=cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)

    return sliced_image


def add_noise(image):
    image = np.clip(image,0,255)
    if bool(random.getrandbits(1)):
        total = IMAGE_WIDTH * IMAGE_HEIGHT * 3
        a = np.random.randint(-3,4, size=total)
        a = a.reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)
        noise = image + a
        to_int_noise = noise.astype('uint8')
    else:
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        to_int_noise = noisy.astype('uint8')
    return to_int_noise


def add_gaussian_noise(image):
    image = np.clip(image, 0, 255)
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    to_int_noisy = noisy.astype('uint8')
    return to_int_noisy


def add_contrast_brightness(image):
    image = np.clip(image,0,255)
    contrast = np.random.uniform(0.9,1.1)
    brightness = np.random.randint(-5,5)
    adjusted_image = image * contrast + brightness
    to_int_adjusted = adjusted_image.astype('uint8')
    return to_int_adjusted


def blur_randomly(image):
    image = np.clip(image, 0, 255)
    std = np.random.uniform(0, 1.5)
    blurred = cv2.GaussianBlur(image, (5,5), std)
    return blurred


def erase_randomly(image):
    area = random.uniform(0.1, 0.6)
    prob = random.getrandbits(1)
    if prob:
        image = Image.fromarray(image)
        w, h = IMAGE_WIDTH, IMAGE_HEIGHT

        w_occlusion_max = int(w * area)
        h_occlusion_max = int(h * area)

        w_occlusion_min = int(w * 0.1)
        h_occlusion_min = int(h * 0.1)

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)

        image.paste(rectangle, (random_position_x, random_position_y))
        image = np.array(image)

    return image


def rotate_image(image, rotation_number):
    """
    Rotate an OpenCV 2 / NumPy image around it's centre by the given angle
    (in degrees). The returned image will have the same size as the new image.

    Adapted from: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    map_coordinate = ROTATE[rotation_number]
    map_x = map_coordinate[0]
    map_y = map_coordinate[1]
    rotate_coin = np.random.randint(3)
    if rotate_coin == 0:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)
    elif rotate_coin == 1:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    else:
        rotated_image=cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
    return rotated_image


def give_random_angle_phi():
    A = np.arange(1,179)
    return random.choice(A)


def generate_rotated_sliced_image(image, rotation_number, sliced_dict, size=None, crop_center=False,
                                    crop_largest_rect=False):
    """
    Generate a valid rotated image for the RotNetDataGenerator. If the
    image is rectangular, the crop_center option should be used to make
    it square. To crop out the black borders after rotation, use the
    crop_largest_rect option. To resize the final image, use the size
    option.
    """

    sliced_image = slice_image(image, sliced_dict)
    rotated_image = rotate_image(sliced_image, rotation_number)

    return rotated_image


def preprocess_image(img_path, color_mode='rgb', flip='True'):

    normalization_layer = Rescaling(1/127.5, offset=-1)
    is_color = int(color_mode == 'rgb')
    image = cv2.imread(img_path, is_color)

    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if bool(random.getrandbits(1)):
        image = add_noise(image)
    if bool(random.getrandbits(1)):
        image = add_contrast_brightness(image)
    if flip:
        if bool(random.getrandbits(1)):
            image = cv2.flip(image, 1)

    return preprocess_input(normalization_layer(image))
