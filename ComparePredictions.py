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
import matplotlib.pyplot as plt
import seaborn as sns

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_Predictions = '../5Predictions/'
path_ProcessedImages= path_ProcessedData + 'ProcessedImages/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'

version = 'V1'
path_YOLOTaggedImagesCurrentVersion = path_ProcessedData + 'YOLOTagged' + version + '/'

filename_ModelPerformance = path_Predictions + 'ModelPerformance.csv'

# %% Compare human tagged to Yolo Version


list_humantags = glob.glob(path_HumanTaggedImages + '*.txt')
list_yolotags = glob.glob(path_YOLOTaggedImagesCurrentVersion + '*.txt')
list_humanimages = glob.glob(path_ProcessedImages + '*.png')

list_humantags_basename = [os.path.basename(name)[:-4] for name in list_humantags]
list_yolotags_basename = [os.path.basename(name)[:-4] for name in list_yolotags]

list_bothtags = list(set(list_humantags_basename).intersection(list_yolotags_basename))
#list_bothtags = list(set(os.path.basename(list_humantags)).intersection(os.path.basename(list_yolotags)))

#list_bothbasenames = [os.path.basename(path)[:-4] for path in list_bothtags]
list_imagenames = [path_ProcessedImages + name + '.png' for name in list_bothtags]

batch_size = 12
objecttype = 0  # 0 is for people

# %% function definitions


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# %%

columns = ['index', 'x_center', 'y_center', 'width', 'height',
                   'confidence', 'other1', 'class']
columns_humantag = ['class', 'x_topleft', 'y_topleft', 'width', 'height']

# %%


#for batch in range(0, len(list_imagenames), batch_size):
#    list_imagenames_current = list_imagenames[batch: batch+batch_size]
#    loaded_images_current = [cv2.imread(imagename) for imagename in list_imagenames_current]
#    for image_index, image in enumerate(loaded_images_current):
#        imagename = list_imagenames_current[image_index]
#
#        humantag = path_HumanTaggedImages + os.path.basename(imagename)[:-4] + '.txt'
#        yolotag = path_YOLOTaggedImagesCurrentVersion + os.path.basename(imagename)[:-4]+ '.txt'
#
#        df_humantag = pd.read_csv(humantag, header=None, delimiter=' ', names=columns_humantag)
#        df_yolotag = pd.read_csv(yolotag, header=0)
#
#        indexer = df_humantag['class'] == 0
#        df_humantag = df_humantag.loc[indexer, :]
#        xscaling = 480.#/416.
#        yscaling = 640. #topleft = int(df_humantag.iloc[0, 1]*xscaling)
#        height = int(df_humantag.iloc[0, 4]*yscaling)
#        width = int(df_humantag.iloc[0, 3]*xscaling)
#        y_centerfromtop = int(df_humantag.iloc[0, 2]*yscaling)
#        x_centerfromleft = int(df_humantag.iloc[0, 1]*xscaling)
#
#        c1, c2 = (int(x_centerfromleft-width/2.), int(y_centerfromtop - height/2)), (int(x_centerfromleft + width/2.), int(y_centerfromtop+height/2.))
#        print(c1, c2)
#        cv2.rectangle(image, c1, c2, (255,0,0), 1)
#
#        c1, c2 = tuple([int(x) for x in df_yolotag.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotag.iloc[0, 3:5]])
#        print(c1, c2)
#        cv2.rectangle(image, c1, c2, (0,255,0), 1)
#        cv2.imshow('image', image)
#        key = cv2.waitKey(2000)

# %%
list_IoU = []
BooleanPrint = False
BooleanShowImages = False

for imagename in list_imagenames:
    image = cv2.imread(imagename)
    # determine detector number from filename
    detectorindex = int(os.path.basename(imagename)[1])
    if BooleanPrint:
        print('\nimage name:', os.path.basename(imagename))

    humantag = path_HumanTaggedImages + os.path.basename(imagename)[:-4] + '.txt'
    yolotag = path_YOLOTaggedImagesCurrentVersion + os.path.basename(imagename)[:-4]+ '.txt'


    df_humantag = pd.read_csv(humantag, header=None, delimiter=' ', names=columns_humantag)
    df_yolotag = pd.read_csv(yolotag, header=0)

    indexer = df_humantag['class'] == 0
    df_humantag = df_humantag.loc[indexer, :]

    indexer = df_yolotag['class'] == 0
    df_yolotag = df_yolotag.loc[indexer, :]

    # we scaled and rotated the detector 1 webcam, so we have
    # to treat them separately.
    if detectorindex == 0:
        xscaling = 480.#/416.
        yscaling = 640. #topleft = int(df_humantag.iloc[0, 1]*xscaling)
    else:
        xscaling = 640
        yscaling = 480

    # Fails if nothing was tagged.
    if len(df_humantag) > 0:
        height = int(df_humantag.iloc[0, 4]*yscaling)
        width = int(df_humantag.iloc[0, 3]*xscaling)
        y_centerfromtop = int(df_humantag.iloc[0, 2]*yscaling)
        x_centerfromleft = int(df_humantag.iloc[0, 1]*xscaling)

    # handle the case where no object was tagged
    if len(df_humantag) > 0:
        c1_1, c2_1 = (int(x_centerfromleft-width/2.), int(y_centerfromtop - height/2)), (int(x_centerfromleft + width/2.), int(y_centerfromtop+height/2.))
        if BooleanPrint:
            print('prediction 1', c1_1, c2_1)
        cv2.rectangle(image, c1_1, c2_1, (255,0,0), 1)

    # handle the case where nothing was predicted.
    if len(df_yolotag) > 0:
        c1_2, c2_2 = tuple([int(x) for x in df_yolotag.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotag.iloc[0, 3:5]])
        if BooleanPrint:
            print('prediction 2', c1_2, c2_2)
        cv2.rectangle(image, c1_2, c2_2, (0,255,0), 1)

    if BooleanShowImages:
        cv2.imshow('image', image)

    IoU = bb_intersection_over_union(list(c1_1) + list(c2_1), list(c1_2) + list(c2_2))
    list_IoU.append(IoU)
    if BooleanPrint:
        print('IoU: {}'.format(IoU))

    if BooleanShowImages:
        key = cv2.waitKey(2000)

# %%

df_ModelPerformance = pd.DataFrame(list_IoU, columns=['V1IoU'], index=list_imagenames)
df_ModelPerformance.to_csv(filename_ModelPerformance)

# %%

fig, ax = plt.subplots()
df_ModelPerformance.hist(ax=ax, bins=50)
ax.set_xlabel('IoU')
ax.set_ylabel('Counts')

# %%
#
#imagename = list_imagenames[0]
## %%
#
#image = cv2.imread(imagename)
#
## determine detector number from filename
#detectorindex = int(os.path.basename(imagename)[1])
#
#humantag = path_HumanTaggedImages + os.path.basename(imagename)[:-4] + '.txt'
#yolotag = path_YOLOTaggedImagesCurrentVersion + os.path.basename(imagename)[:-4]+ '.txt'
#
#
#df_humantag = pd.read_csv(humantag, header=None, delimiter=' ', names=columns_humantag)
#df_yolotag = pd.read_csv(yolotag, header=0)
#
#indexer = df_humantag['class'] == 0
#df_humantag = df_humantag.loc[indexer, :]
#
#indexer = df_yolotag['class'] == 0
#df_yolotag = df_yolotag.loc[indexer, :]
#
## we scaled and rotated the detector 1 webcam, so we have
## to treat them separately.
#if detectorindex == 0:
#    xscaling = 480.#/416.
#    yscaling = 640. #topleft = int(df_humantag.iloc[0, 1]*xscaling)
#else:
#    xscaling = 640
#    yscaling = 480
#
#height = int(df_humantag.iloc[0, 4]*yscaling)
#width = int(df_humantag.iloc[0, 3]*xscaling)
#y_centerfromtop = int(df_humantag.iloc[0, 2]*yscaling)
#x_centerfromleft = int(df_humantag.iloc[0, 1]*xscaling)
#
#c1, c2 = (int(x_centerfromleft-width/2.), int(y_centerfromtop - height/2)), (int(x_centerfromleft + width/2.), int(y_centerfromtop+height/2.))
#print(c1, c2)
#cv2.rectangle(image, c1, c2, (255,0,0), 1)
#
#c1, c2 = tuple([int(x) for x in df_yolotag.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotag.iloc[0, 3:5]])
#print(c1, c2)
#cv2.rectangle(image, c1, c2, (0,255,0), 1)
#cv2.imshow('image', image)
#key = cv2.waitKey(10000)