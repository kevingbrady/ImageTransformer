import os
import pandas as pd
import cv2
import matplotlib
import numpy as np
from utils import showImage


if __name__ == '__main__':

    file_list = [x for x in os.listdir('Chip_Thorlabs_100_um') if x.endswith('.csv')]
    #image_frame = pd.read_csv('Chip_Thorlabs_100_um/' + file_list[138])
    #img = image_frame.to_numpy(dtype=np.uint8)
    #print(img.shape)
    #showImage(img)

    for i in file_list:

        image_frame = pd.read_csv('Chip_Thorlabs_100_um/' +   i)
        img = image_frame.to_numpy(dtype=np.uint8)

        print(img.shape)
        #showImage(img)
