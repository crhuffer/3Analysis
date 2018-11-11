# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:35:24 2018

@author: crhuffer
"""

import time
import cv2
import datetime
import numpy as np
# %% Setup paths

images = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"

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
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    if ret:

        filename1 = (path_ProcessedData +
                    'D{}Date{}.png'.format(DeviceID1,
                      datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
                    )
        print(filename1)
        cv2.imwrite(filename1, frame1)

        filename2 = (path_ProcessedData +
                    'D{}Date{}.png'.format(DeviceID2,
                      datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
                    )
        print(filename2)
        cv2.imwrite(filename2, frame2)

#        cv2.imshow("frame", np.concatenate((frame1, frame2), axis=1))
        cv2.imshow('frame', frame1)
        key = cv2.waitKey(1)
