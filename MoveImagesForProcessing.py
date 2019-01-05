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

def MoveImagesForProcessing(list_ImagesToProcess, path_ImagesToProcess):
    for index, imagename in enumerate(list_ImagesToProcess):
        filename = os.path.basename(imagename)
        destination = path_ImagesToProcess + filename
        shutil.copyfile(imagename, destination)
        if index%100 == 0:
            print('Moved image {} of {} images'.format(index,
                  len(list_ImagesToProcess)))
    
#     %% Delete the files form NewImages if we were successful (made it this far)
    
    for index, imagename in enumerate(list_ImagesToProcess):
        os.remove(imagename)
        if index%100 == 0:
            print('Removed image {} of {} images'.format(index,
                  len(list_ImagesToProcess)))
            
# %%

# %%

#path_Predictions = path_ProcessedData + 'YOLOTaggedV1/'        
def GetMissedImagesForProcessing(path_ProcessedImages, path_Predictions, path_ImagesToProcess):
    ''' Compare files in processedimages and predictions to find images that
    have not been processed and return them.
    '''
    
    
    list_FilesFromProcessedImages = glob.glob(path_ProcessedImages + '*.png')
    list_FilesFromPredictions = glob.glob(path_Predictions + '*.txt')
    
    series_NamesFromProcessedImages = pd.Series(list_FilesFromProcessedImages).apply(lambda x: os.path.basename(x)[:-4])
    series_NamesFromPredictions = pd.Series(list_FilesFromPredictions).apply(lambda x: os.path.basename(x)[:-4])
    
    indexer = ~series_NamesFromProcessedImages.isin(series_NamesFromPredictions)
#    series_NameMissingPredictions = series_NamesFromProcessedImages.loc[indexer].apply(lambda x: path_ImagesToProcess + x + '.png')
#    list_NameMissingPredictions = list(series_NameMissingPredictions)
#    list_NameMissingPredictions = list(series_NamesFromProcessedImages.loc[indexer])
    list_NameMissingPredictions = list(series_NamesFromProcessedImages.loc[indexer].apply(lambda x: path_ProcessedImages + x + '.png'))
    return list_NameMissingPredictions

# %%

#GetMissedImagesForProcessing(path_ProcessedImages, path_Predictions, path_ImagesToProcess)

# %%

if '__name__' == '__main__':
             
    MoveImagesForProcessing(list_ImagesToProcess, path_ImagesToProcess)


    
    