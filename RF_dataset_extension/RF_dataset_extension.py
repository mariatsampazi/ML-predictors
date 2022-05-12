# The csv files used can be found in the following path: https://github.com/wineslab/colosseum-oran-commag-dataset/blob/main/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1
# The base for the code can be found here: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# The purpose of this code is to extend the dataset with the use of the Random Forest (RF) algorithm.
# For the testing of the accuracy of the RF, we use the RF_testing_accuracy.py code.

from cgi import test
from sqlite3 import Timestamp
import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt
import io
import os
import glob
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, RepeatedKFold,  train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime

#------------------------------------------------------------------------------------
#We set working directory and we find all the csv files in the folder.
# The glob pattern matching is -> extension ='csv'. Finally, we save
# all the results in a list, namely: all_filenames. The encoding used is: utf-8.

os.chdir("/Users/maria/Documents/GitHub/oran_efforts/slices_bs1")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# -- uncomment the following line for debugging purposes --
#print(all_filenames) 

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#Export to csv. This csv contains all the metrics for this path:
# https://github.com/wineslab/colosseum-oran-commag-dataset/tree/main/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1
# i.e. the metrics for: colosseum-oran-commag-dataset/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1/
os.chdir("/Users/maria/Documents/GitHub/oran_efforts/RF_dataset_extension/combined_csv_file")
combined_csv.to_csv( "exp1_slices_bs1_CombinedCSV.csv", index=False, encoding='utf-8-sig')

# With the following lines we read the data (the data is in tidy data format),
# and we remove empty columns (i.e. Unnmaned column names), since the Random Forest Algorithm
# will later complain about the NaN entries.
# Features: contain the actual data of our dataset. All the knowned info we know about the slices of our BS 1.
features=pd.read_csv("exp1_slices_bs1_CombinedCSV.csv") # actual data
features = features.loc[:, ~features.columns.str.contains('^Unnamed')]

# -- uncomment the following line for debugging purposes to print the size of the input -- 
#print('The shape of our features is:', features.shape)
# -- uncomment the following lines to check if the .csv file is read correctly. The check is done with the first 5 entries --
#print(features.head(5))
# With the following line we will print some descriptive statistics for each column, for the first 5 entries of the dataset.
#print(features.describe()) 

#print(features.head(5))
#features.sort_values(by=['Timestamp'])
#print(features.head(5))

# x1000 scaling of the input
features=1000*features
#Uncomment the following line for debugging purposes
#print(features.shape)

#'''
# Labels are the values we want to predict, we want to predict the downlink throughput
labels=np.array(features['tx_brate downlink [Mbps]'])
# Handle empty entries, otherwise the algorithm will complain. The following line will be commented because line 48 takes care of the aforementioned problem.
#labels = np.nan_to_num(labels) 

# Remove the labels from the features. Axis 1 refers to the columns.
features= features.drop('tx_brate downlink [Mbps]', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
#print(feature_list)

# Convert to numpy array
features = np.array(features)
# Handle empty entries, otherwise the algorithm will complain. The following line will be commented because line 48 takes care of the aforementioned problem.
#features= np.nan_to_num(features)
# -- Uncomment the following line for debugging purposes --
#print(features)

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- Training & Testing Sets -------------------------------------------------------------------

# Split the data into training and testing sets. State is set to 42 for reproducible results i.e, the results will be the same each time I run the split.
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=0.25, random_state = 42)
'''
#   ------------------------------------------------- uncomment the following lines for debugging purposes -----------------------------------------------
#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)
# --------------------------------------------------------------------------------------------------------------------------------------------------------
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------Random Forest Algorithm-----------------------------------------------------------------------
# n_estimators: indicates the number of decision trees used
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42,max_features='log2', n_jobs=-1)
rf = RandomForestRegressor(n_estimators = 10, random_state = 42, n_jobs=-1)
# Train the model on training data (75% of the whole dataset)
rf.fit(train_features, train_labels)
#predictions = rf.predict(features[1:100])
predictions = rf.predict(features)
#print(predictions[1000])
#print(labels[1000])
#print((predictions[1000]-labels[1000])*0.001)

# (1) Use the forest's predict method on the whole dataset
##predictions = rf.predict(test_features)
#Uncomment the following line for debugging purposes
#print(predictions.shape)
#print(type(features))
#print(type(predictions))

row = features[-600:].flatten()
# make a one-step prediction

#print(predictions)
#print(test_features.shape)

#'''
#---------------------------------------------------actual timestamps--------------------------------------------------
# From an observation of the timestamps on the dataset of the existing csv files, we observe that all the timestamps
# have an offset of 250, between them. Also, if we check the combined csv file, again all the timestamps have an offset
# of 250 between them.

# actual timestamps from the dataset
timestamps = features[:, feature_list.index('Timestamp')]
#--Uncomment the following line for debugging purposes--
#print(len(timestamps))
real_timestamps=timestamps
#print(real_timestamps[-1])
#print(type(real_timestamps))
# save the value of the last (actual) timestamp from the combined csv file
#the following lines give the same result
#final_actual_timestamp2= real_timestamps[-1]
final_actual_timestamp= max(real_timestamps)

#--Uncomment the following line for debugging purposes--
#print(final_actual_timestamp)
#print(final_actual_timestamp2)

#-------------------------------------------Creation of the next timestamps----------------------------------------------
#                                        *****
# Save the last timestamp in order to create the next timestamps.
# Our input is scaled by x1000. So the timestamps are also scaled by x1000.
#                                       ----
# We want the number of the predicted timestamps to match the number of the predictions, if we do (1).
# Our dataset has a specific size, i.e. (a rows, b columns), and given the fact that our predictions
# are based on the (entire input) dataset, their size will also be (a,b).
#                                       ----
# operation of range function: range(1, 5) -> returns 1,2,3,4 (not 5)
# if we have a set of values values =[8, 10, 12, 14], with offeset_index=2, we obtain 14 as follows: 8 + 2*3,
# i.e., fist(values)+offset_index*[size(values)-1], if we to obtain any other intermediate value we again follow
# a similar pattern, i.e. 10 = 8 + 2*1, where 1 is again size(values)-1, when the stack (array) has until this
# point 2 entries.

next_timestamps = []

# observed offset is 250, but input is scaled by x1000 -> 250*1000=250000
timestamp_offset=250000
# fix the float to int problem 
final_actual_timestamp=int(final_actual_timestamp)
#print(final_actual_timestamp)
last_predicted_timestamp=final_actual_timestamp + timestamp_offset*(len(predictions)-1)
# fix the float to int problem 
last_predicted_timestamp=int(last_predicted_timestamp)
#print(last_predicted_timestamp)

first_expected_timestamp=int(final_actual_timestamp+timestamp_offset)
#--Uncomment the following line for debugging purposes--
#print(first_expected_timestamp-final_actual_timestamp)

for x in range(first_expected_timestamp, last_predicted_timestamp+2*timestamp_offset, timestamp_offset):
  next_timestamps.append(x)
  
# convert list to numpy array for consistency reasons with the other variables
next_timestamps= np.array(next_timestamps)
next_timestamps = next_timestamps. astype(float)

# Uncomment the following lines for debugging purposes
#print(next_timestamps[0]-first_expected_timestamp)
#print(first_expected_timestamp-final_actual_timestamp)
#print(first_expected_timestamp)
#print(final_actual_timestamp)

#print(next_timestamps[0])

#Uncomment the following line for debugging purposes
#print(last_predicted_timestamp-next_timestamps[1])
#print(last_predicted_timestamp)

#Uncomment the following line for debugging purposes
#print(type(next_timestamps))
#print(type(timestamps))
#print(len(next_timestamps))
#print(timestamps[-1])
#print(next_timestamps[0])
#print(predictions)
#print(predictions.shape)
#print(features.shape)
'''
'''
#----------------------------------------------------------------------------------------
# find the actual throughput downlink values for the actual timestamps
times=[str(int(Timestamp)) for Timestamp in timestamps]
true_data = pd.DataFrame(data = {'time': times, 'actual': labels})
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
# find the predicted throughput downlink values for the predicted timestamps
next_times = [str(int(Timestamp))  for Timestamp in next_timestamps]
predictions_data = pd.DataFrame(data = {'time': next_times, 'prediction': predictions})
#----------------------------------------------------------------------------------------
#--Uncomment the following lines for debugging purposes--
#print(len(true_data['actual']))
#print(len(predictions_data['prediction']))
#--------------------------------------------------------

# save values for plotting in matlab
real=true_data['actual']
pred=predictions_data['prediction']

#--Uncomment the following lines for debugging purposes--
#print(real)
#print(pred)
#--------------------------------------------------------

# reset path again for the saving of the excel files
os.chdir("/Users/maria/Documents/GitHub/oran_efforts/RF_dataset_extension/results_and_plotting")

# save the actual and predicted downlink throughput
real.to_csv('real.csv', index = False,header=False)
pred.to_csv('next_pred.csv', index = False,header=False)

# save the actual timestamps from the datasets and timestamps from the testing set
df = pd.DataFrame(real_timestamps)
df.to_csv('real_time.csv', index=False,header=False)
df2 = pd.DataFrame(next_timestamps)
df2.to_csv('next_pred_time.csv', index=False,header=False)

# close all plots
plt.close('all')

#'''