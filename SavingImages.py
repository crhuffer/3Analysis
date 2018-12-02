# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:35:24 2018

@author: crhuffer
"""

import time
import cv2
import datetime
import numpy as np

# turns off
BooleanWillImagesSave = False

# %% Setup paths

images = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_NewImages = path_ProcessedData + 'NewImages/'

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

# %% Setup input device

DeviceID1 = 0
cap1 = cv2.VideoCapture(DeviceID1)
assert cap1.isOpened(), 'Cannot capture source'

DeviceID2 = 1
cap2 = cv2.VideoCapture(DeviceID2)
assert cap2.isOpened(), 'Cannot capture source'

# %% Capture, display, and save frames from video source
while cap1.isOpened():
    time.sleep(0.05)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        now = datetime.datetime.now()
        formatstr = '%Y%m%d%H%M%S%f'
        filename1 = (path_NewImages +
                    'D{}Date{}.png'.format(DeviceID1,
                      now.strftime(formatstr))
                    )
        print(filename1)
#        cv2.imwrite(filename1, frame1)

        filename2 = (path_NewImages +
                    'D{}Date{}.png'.format(DeviceID2,
                      now.strftime(formatstr))
                    )
        print(filename2)
#        cv2.imwrite(filename2, frame2)

#        cv2.imshow("frame", np.concatenate((frame1, frame2), axis=1))
#        cv2.imshow("frame", np.concatenate((frame1, cv2.getRotationMatrix2D(frame2, 90, 1)), axis=1))
        angle=90
        frame1_rotated = rotate_bound(frame1, angle)

        angle=0
        frame2_rotated = rotate_bound(frame2, angle)
        if BooleanWillImagesSave:
            cv2.imwrite(filename1, frame1_rotated)
            cv2.imwrite(filename2, frame2_rotated)

#        print(frame1_rotated.shape, frame2_rotated.shape)
#        cv2.imshow("frame", np.concatenate((cv2.getRotationMatrix2D(frame1, 90, 1), cv2.getRotationMatrix2D(frame2, 90, 1)), axis=1))
#        cv2.imshow("frame", np.concatenate((frame1_rotated, frame2_rotated), axis=1))
        cv2.imshow("frame", np.concatenate((frame1_rotated, np.pad(frame2_rotated, pad_width=((0, 160), (0, 0), (0, 0)), mode='minimum')), axis=1))
#        cv2.imshow('frame', frame1)
        key = cv2.waitKey(1)
