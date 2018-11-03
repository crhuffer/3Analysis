# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:35:24 2018

@author: crhuffer
"""

from __future__ import division

import time
import os
import os.path as osp
import numpy as np
import torch
import cv2
import pickle as pkl
from utilis import load_classes, parse_cfg, get_test_input, create_module, letterbox_image, prep_image, filter_results
import random
from yolo_v3 import yolo_v3

#import pandas as pd
import datetime


#layer_type_dic, module_list = create_module(blocks)
# it might imporve performance
#torch.backends.cudnn.benchmark = True

# change cuda to True if you wish to use gpu
cuda = False
images = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
batch_size = 1
confidence = float(0.7)
nms_thesh = float(0.4)
start = 0
num_classes = 80
cfg_path = "../4Others/yolov3.cfg"
blocks = parse_cfg(cfg_path)
model = yolo_v3(blocks)
model.load_weights("../4Weights/yolov3.weights")
colors = pkl.load(open("../4Others/pallete", "rb"))
if cuda:
    model = model.cuda()
for params in model.parameters():
    params.requires_grad = False
model.layer_type_dic['net_info']["height"] = 320
model.layer_type_dic['net_info']["width"] = 320
inp_dim = int(model.layer_type_dic['net_info']["height"])


DeviceID = 0
cap = cv2.VideoCapture(DeviceID)
assert cap.isOpened(), 'Cannot capture source'

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
#        img = prep_image(frame, inp_dim)
        filename = path_ProcessedData + 'D{}Date{}.png'.format(DeviceID, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
#        print(filename)
        cv2.imwrite(filename, frame)
        
    