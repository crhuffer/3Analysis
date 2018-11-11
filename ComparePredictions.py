# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:20:41 2018

@author: Craig
"""

# %% Library imports

import cv2
import glob
import os
import pandas as pd

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"

path_ProcessedImages= path_ProcessedData + 'ProcessedImages/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'

version = 'V1'
path_YOLOTaggedImagesCurrentVersion = path_ProcessedData + 'YOLOTagged' + version + '/'

# %% Compare human tagged to Yolo Version


list_humantags = glob.glob(path_HumanTaggedImages + '*.txt')
list_yolotags = glob.glob(path_YOLOTaggedImagesCurrentVersion + '*.txt')
list_humanimages = glob.glob(path_ProcessedImages + '*.png')

list_bothtags = list(set(list_humantags).intersection(list_humantags))

list_bothbasenames = [os.path.basename(path)[:-4] for path in list_bothtags]
list_imagenames = [path_ProcessedImages + name + '.png' for name in list_bothbasenames]

batch_size = 12

for batch in range(0, len(list_imagenames), batch_size):
    list_imagenames_current = list_imagenames[batch: batch+batch_size]
    loaded_images_current = [cv2.imread(imagename) for imagename in list_imagenames_current]
    for image_index, image in enumerate(loaded_images_current):
        imagename = list_imagenames_current[image_index]

        humantag = path_HumanTaggedImages + os.path.basename(imagename)[:-4] + '.txt'
        yolotag = path_YOLOTaggedImagesCurrentVersion + os.path.basename(imagename)[:-4]+ '.txt'
        columns = ['index', 'x_center', 'y_center', 'width', 'height',
                   'confidence', 'other1', 'class']
        df_humantag = pd.read_csv(humantag, header=None, delimiter=' ')
        df_yolotag = pd.read_csv(yolotag, header=0)

        xscaling = 640.#/416.
        yscaling = 480. #topleft = int(df_humantag.iloc[0, 1]*xscaling)
        height = int(df_humantag.iloc[0, 4]*yscaling)
        width = int(df_humantag.iloc[0, 3]*xscaling)
        y_centerfromtop = int(df_humantag.iloc[0, 2]*yscaling)
        x_centerfromleft = int(df_humantag.iloc[0, 1]*xscaling)

        c1, c2 = (int(x_centerfromleft-width/2.), int(y_centerfromtop - height/2)), (int(x_centerfromleft + width/2.), int(y_centerfromtop+height/2.))
        print(c1, c2)
        cv2.rectangle(image, c1, c2, (255,0,0), 1)

        c1, c2 = tuple([int(x) for x in df_yolotag.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotag.iloc[0, 3:5]])
        print(c1, c2)
        cv2.rectangle(image, c1, c2, (0,255,0), 1)
        cv2.imshow('image', image)
        key = cv2.waitKey(2000)