# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:00:36 2018

@author: Craig
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_Predictions = "../5Predictions/"

filename_Predictions = path_Predictions + 'Predictions - Copy.xlsx'

version = 'V1'
path_YOLOTaggedImagesCurrentVersion = path_ProcessedData + 'YOLOTagged' + version + '/'

# %%

list_predictionfilenames = glob.glob(path_YOLOTaggedImagesCurrentVersion + '*.txt')

# %%

list_datetimes = []
list_detector = []
df_Predictions = pd.DataFrame()

for filename in list_predictionfilenames:


    # Get the values from the file and add them to what we have
    df_Prediction = pd.read_csv(filename)

    # Grab the datetime and detector number from the filename
    current_datetime = datetime.datetime.strptime(filename[-24:-4], '%Y%m%d%H%M%S%f')
    current_detector = int(filename[-29:-28])

    # Add the current values for however many predictions are in this frame.
    list_datetimes.extend([current_datetime]*len(df_Prediction))
    list_detector.extend([current_detector]*len(df_Prediction))
    ## TODO: Extract the datetime and detector from the filename, add as a new column.
    if len(df_Predictions) == 0:
        df_Predictions = df_Prediction.copy()
    else:
        df_Predictions = df_Predictions.append(df_Prediction, ignore_index=True)

# add the detectors and datetimes to the dataframe
df_Predictions['detector'] = list_detector
df_Predictions['datetime'] = list_datetimes

# %%

df_Predictions.to_excel(filename_Predictions)

#df_Predictions = pd.read_csv(filename_Predictions, index_col=0)

# %%

date_format = '%Y%m%d%H%M%S%f'
df_Predictions['datetime'] = pd.to_datetime(df_Predictions['filename'].apply(lambda x: x[-24:-4]), format=date_format)
df_Predictions.head()

# %%

fig, ax = plt.subplots()
plt.plot_date(data = df_Predictions, x='datetime', y='height', marker='.')

# %%

df_Predictions1 = df_Predictions[:1300]

# %%
fig, ax = plt.subplots(3, 1, figsize = (25,12))
label = 'height'; index_ax = 0; color = 'r'
plt.sca(ax[index_ax]); ax[index_ax].set_ylabel(label); plt.grid()
plt.plot_date(data = df_Predictions1, x='datetime', y=label, marker='.', color=color, label=label)
label = 'width'; index_ax = 1; color = 'g'
plt.sca(ax[index_ax]); ax[index_ax].set_ylabel(label); plt.grid()
plt.plot_date(data = df_Predictions1, x='datetime', y=label, marker='.', color=color, label=label)
label = 'y_center'; index_ax = 2; color = 'b'
plt.sca(ax[index_ax]); ax[index_ax].set_ylabel(label); plt.grid()
plt.plot_date(data = df_Predictions1, x='datetime', y=label, marker='.', color=color, label=label)
fig.legend()

ax[2].set_xlabel('datetime')