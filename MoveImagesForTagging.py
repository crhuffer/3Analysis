# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:55:22 2018

@author: Craig
"""
# %% Library imports

import glob
import pandas as pd
import random
import shutil
import os
import zipfile

# %% Setup variables
# the number of images to move into the totag folder
imagestoprocess = 10

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_Downloads = "../../../../Downloads/"

path_ImagesToTag= path_ProcessedData + 'ImagesToTag/'
path_ProcessedImages= path_ProcessedData + 'ProcessedImages/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'

# %% Move the predictions to the human tagged folder

filename_boxes = path_Downloads + 'bboxes_yolo.zip'
boxes_zip = zipfile.ZipFile(filename_boxes)
boxes_zip.extractall(path_HumanTaggedImages)
boxes_zip.close()

os.remove(filename_boxes)

# %% Delete the images in the to tag folder

list_filestodelete = glob.glob(path_ImagesToTag + "*.png")
for filename in list_filestodelete:
    os.remove(filename)

# %% Determine a list of nontagged images

list_humantags = glob.glob(path_HumanTaggedImages + '*.txt')
list_humantags_base = list(pd.Series(list_humantags).apply(lambda x: os.path.basename(x[:-4])))
list_humanimages = glob.glob(path_ProcessedImages + '*.png')
list_humanimages_base = list(pd.Series(list_humanimages).apply(lambda x: os.path.basename(x[:-4])))
list_nontagged = list(set(list_humanimages_base)-set(list_humantags_base))

# %% Randomly sample some of the nontagged images

list_toprocess = pd.Series(random.sample(list_nontagged, imagestoprocess)).apply(lambda x: path_ProcessedImages + x+'.png')

# %% Move the determined images to the totag folder

for imagename in list_toprocess:
    filename = os.path.basename(imagename)
    destination = path_ImagesToTag + filename
    shutil.copyfile(imagename, destination)

