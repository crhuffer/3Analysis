# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:55:22 2018

@author: Craig
"""
# %% Library imports

import glob
import pandas as pd
import shutil
import os

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_Downloads = "../../../../Downloads/"

path_ProcessedImages = path_ProcessedData + 'ProcessedImages/'
path_NewImages = path_ProcessedData + 'NewImages/'
path_ImagesToProcess = path_ProcessedData + 'ImagesToProcess/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'

# %% Move files from NewImages to ImagesToProcess

list_ImagesToProcess = glob.glob(path_NewImages+'*.png')
for index, imagename in enumerate(list_ImagesToProcess):
    filename = os.path.basename(imagename)
    destination = path_ImagesToProcess + filename
    shutil.copyfile(imagename, destination)
    if index%100 == 0:
        print('Moved image {} of {} images'.format(index,
              len(list_ImagesToProcess)))

# %% Delete the files form NewImages if we were successful (made it this far)
for index, imagename in enumerate(list_ImagesToProcess):
    os.remove(imagename)
    if index%100 == 0:
        print('Removed image {} of {} images'.format(index,
              len(list_ImagesToProcess)))
