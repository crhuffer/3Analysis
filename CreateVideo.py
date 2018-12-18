# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:43:23 2018

@author: crhuffer
"""

import pandas as pd
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# %% Versions

version = 'V1'

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_NewImages= path_ProcessedData + 'NewImages/'
path_ImagesToProcess= path_ProcessedData + 'ImagesToProcess/'
path_ProcessedImages= path_ProcessedData + 'ProcessedImages/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'
path_YOLOTaggedImagesCurrentVersion = path_ProcessedData + 'YOLOTagged' + version + '/'
path_Predictions = "../5Predictions/"


# %%

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# %% Get all of the processed images

list_imagenamesfull = glob.glob(path_ProcessedImages + '*.png')
list_imagenames = list(pd.Series(list_imagenamesfull).apply(lambda x: os.path.basename(x)))

# %% Compare human tagged to Yolo Version


list_humantags = glob.glob(path_HumanTaggedImages + '*.txt')
list_yolotags = glob.glob(path_YOLOTaggedImagesCurrentVersion + '*.txt')
list_humanimages = glob.glob(path_ProcessedImages + '*.png')

list_humantags_basename = [os.path.basename(name)[:-4] for name in list_humantags]
list_yolotags_basename = [os.path.basename(name)[:-4] for name in list_yolotags]


columns = ['index', 'x_center', 'y_center', 'width', 'height',
                   'confidence', 'other1', 'class']
columns_humantag = ['class', 'x_topleft', 'y_topleft', 'width', 'height']

# %% Grab the detector number and datetime

df_images = pd.DataFrame(list_imagenames, columns=['imagenames'])
df_images['filename'] = list_imagenamesfull
df_images['detector'] = df_images['imagenames'].apply(lambda x: int(x[1]))
df_images['datetime'] = pd.to_datetime(df_images['imagenames'].apply(lambda x: x[6: -4]), format='%Y%m%d%H%M%S%f')

# %% Sort the values

df_images.sort_values(by=['datetime', 'detector'], inplace=True)

# %% Make a timedelta

#df_images['timedelta'] = 0

df_images['timedelta'] = df_images['datetime'] - df_images['datetime'].shift()

# %% Plot the timedelta on different scales

# meant to show the time between sessions, time between lists, and
# the time between frames. Shows that frames seem to come in faster than 1s.

indexer = df_images['detector'] == 0
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(12,5))
plt.sca(ax[0])
ax[0].set_yscale('log')
df_images.loc[indexer, 'timedelta'].dt.days.hist(bins=50)
ax[0].set_xlabel('days')
ax[0].set_ylabel('Counts')
plt.sca(ax[1])
ax[1].set_xlabel('s')
df_images.loc[indexer, 'timedelta'].dt.seconds.hist(bins=np.linspace(0, 3600, 50))
plt.sca(ax[2])
ax[2].set_xlabel('s')
df_images.loc[indexer, 'timedelta'].dt.seconds.hist(bins=np.linspace(0, 500, 50))
plt.sca(ax[3])
df_images.loc[indexer, 'timedelta'].dt.microseconds.hist(bins=np.linspace(0, 2000000, 50))
ax[3].set_xlabel('us')

# %% determined that the threshold for a session should be 2 seconds

indexer = df_images['timedelta'] > pd.datetools.timedelta(seconds=2)
df_images['session_index'] = indexer.cumsum()

# %% group the dataframe into sessions

groups = df_images.groupby('session_index')

# %% Calculate properties of the sessions

list_sessiondurations = []
list_sessionframes = []
list_sessionstart = []
list_sessionend = []
for session_index, df_session in groups:
    df_session.describe()
    list_sessionstart.append(df_session['datetime'].min())
    list_sessionend.append(df_session['datetime'].max())
    list_sessiondurations.append(df_session['datetime'].max() - df_session['datetime'].min())
    list_sessionframes.append(len(df_session))

# %% Save the properties of the sessions to a sessions dataframe

df_sessions = pd.DataFrame(list_sessiondurations, columns=['duration'])
df_sessions['frames'] = list_sessionframes
df_sessions['datetime_start'] = list_sessionstart
df_sessions['datetime_end'] = list_sessionend

# %%

indexer = df_session['detector'] == 0
list_filenamesD0 = list(df_session.loc[indexer, 'filename'])

indexer = df_session['detector'] == 1
list_filenamesD1 = list(df_session.loc[indexer, 'filename'])

# %%

for index in range(len(list_filenamesD0)):
    filenameD0 = list_filenamesD0[index]
    imgD0 = cv2.imread(filenameD0)
    filenameD1 = list_filenamesD1[index]
    imgD1 = cv2.imread(filenameD1)


#    humantag = path_HumanTaggedImages + os.path.basename(imagename)[:-4] + '.txt'
    yolotagD0 = path_YOLOTaggedImagesCurrentVersion + os.path.basename(filenameD0)[:-4]+ '.txt'
    yolotagD1 = path_YOLOTaggedImagesCurrentVersion + os.path.basename(filenameD1)[:-4]+ '.txt'

    df_yolotagD0 = pd.read_csv(yolotagD0, header=0)
    indexer = df_yolotagD0['class'] == 0
    df_yolotagD0 = df_yolotagD0.loc[indexer, :]

    # handle the case where nothing was predicted.
    if len(df_yolotagD0) > 0:
        c1_2, c2_2 = tuple([int(x) for x in df_yolotagD0.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotagD0.iloc[0, 3:5]])
        cv2.rectangle(imgD0, c1_2, c2_2, (0,255,0), 1)

    df_yolotagD1 = pd.read_csv(yolotagD1, header=0)
    indexer = df_yolotagD1['class'] == 0
    df_yolotagD1 = df_yolotagD1.loc[indexer, :]

    # handle the case where nothing was predicted.
    if len(df_yolotagD1) > 0:
        c1_2, c2_2 = tuple([int(x) for x in df_yolotagD1.iloc[0, 1:3]]), tuple([int(x) for x in df_yolotagD1.iloc[0, 3:5]])
        cv2.rectangle(imgD1, c1_2, c2_2, (0,255,0), 1)

    imgD1 = np.pad(imgD1, pad_width=((0, 160), (0, 0), (0, 0)), mode='minimum')
#    angle=90
#    frame1_rotated = rotate_bound(imgD0, angle)
#
#    angle=0
#    frame2_rotated = rotate_bound(imgD1, angle)
#    if BooleanWillImagesSave:
#            cv2.imwrite(filename1, frame1_rotated)
#            cv2.imwrite(filename2, frame2_rotated)

#        print(frame1_rotated.shape, frame2_rotated.shape)
#        cv2.imshow("frame", np.concatenate((cv2.getRotationMatrix2D(frame1, 90, 1), cv2.getRotationMatrix2D(frame2, 90, 1)), axis=1))
#        cv2.imshow("frame", np.concatenate((frame1_rotated, frame2_rotated), axis=1))

    cv2.imshow("frame", np.concatenate((imgD0, imgD1), axis=1))
#        cv2.imshow('frame', frame1)
    key = cv2.waitKey(1)

