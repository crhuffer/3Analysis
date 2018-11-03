# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:35:24 2018

@author: crhuffer
"""

import time
import cv2
import datetime

# %% Setup paths

images = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"

# %% Setup input device

DeviceID = 0
cap = cv2.VideoCapture(DeviceID)
assert cap.isOpened(), 'Cannot capture source'

# %% Capture, display, and save frames from video source
while cap.isOpened():
    time.sleep(0.05)
    ret, frame = cap.read()

    if ret:

        filename = (path_ProcessedData +
                    'D{}Date{}.png'.format(DeviceID,
                      datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
                    )
        print(filename)
        cv2.imwrite(filename, frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
