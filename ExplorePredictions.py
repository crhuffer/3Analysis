# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:00:36 2018

@author: Craig
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Setup paths

path_RawData = "../1RawData/"
path_ProcessedData = "../2ProcessedData/"
path_Predictions = "../5Predictions/"

filename_Predictions = path_Predictions + 'Predictions - Copy.csv'

# %%

df_Predictions = pd.read_csv(filename_Predictions, index_col=0)

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