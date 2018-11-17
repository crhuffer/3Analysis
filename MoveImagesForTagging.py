# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:55:22 2018

@author: Craig
"""
# %%

import glob
import pandas as pd
import random
import shutil
import os

# %%

imagestoprocess = 20

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"

path_ImagesToProcess= path_ProcessedData + 'ImagesToProcess/'
path_ProcessedImages= path_ProcessedData + 'ProcessedImages/'
path_HumanTaggedImages = path_ProcessedData + 'HumanTagged/'

# %%

list_humantags = glob.glob(path_HumanTaggedImages + '*.txt')
list_humantags_base = list(pd.Series(list_humantags).apply(lambda x: x[:-4]))
list_humanimages = glob.glob(path_ProcessedImages + '*.png')
list_humanimages_base = list(pd.Series(list_humanimages).apply(lambda x: x[:-4]))
list_nontagged = list(set(list_humanimages_base)-set(list_humantags_base))

# %%

list_toprocess = pd.Series(random.sample(list_nontagged, imagestoprocess)).apply(lambda x: x+'.png')

# %%

for imagename in list_toprocess:
    filename = os.path.basename(imagename)
    destination = path_ImagesToProcess + filename
    shutil.copyfile(imagename, destination)

