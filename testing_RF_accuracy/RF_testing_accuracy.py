# Python code used as example for data manipulation.
# The csv files used for this example can be found in the following path: https://github.com/wineslab/colosseum-oran-commag-dataset/blob/main/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1
# The base for the code can be found here: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# The purpose of this code is to:
# a) plot some metrics from the csv files (no8 metrics file)
# b) test the accuracy of the Random Forest Algorithm (RF) by comparing the predictions to a known test
# c) optimizing the efficiency of the RF
# d) This code is NOT used for the extension of the dataset

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
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

#------------------------------------------------------------------------------------
#We set working directory and we find all the csv files in the folder.
# The glob pattern matching is -> extension ='csv'. Finally, we save
# all the results in a list, namely: all_filenames. The encoding used is: utf-8.
# The csv files are not combined in order (i.e. 1,2,3,...)

os.chdir("/Users/maria/Documents/GitHub/oran_efforts/slices_bs1")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# -- uncomment the following line for debugging purposes --
#print(all_filenames) 

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#  -- Uncomment the following line to print the current working directory for debugging purposes --
#print(os.getcwd())  

#Export to csv. This csv contains all the metrics for this path:
# https://github.com/wineslab/colosseum-oran-commag-dataset/tree/main/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1
# i.e. the metrics for: colosseum-oran-commag-dataset/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1/
os.chdir("/Users/maria/Documents/GitHub/oran_efforts/testing_RF_accuracy/combined_csv_file")
combined_csv.to_csv( "exp1_slices_bs1_CombinedCSV.csv", index=False, encoding='utf-8-sig')
#print(os.getcwd()) 
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

# ** PLOTTING **
#  **** --------------------------------------------------------------------------------------------- ****
# The following lines address datasets which are incomplete, i.e. the contain zero entries. Our
# datasets (i.e. metric csv files) contain 0 entries, but 0s indicate that the users do not transmit 
# at a specific timestamp, not a lack of data. So the following lines have no application in our case,
# and they are included for completeness reasons.

# replace 0 values with NaN, used later for plotting! Not needed for our case.
#features.replace(0, np.nan,inplace=True) 
#features['dl_mcs'] = features['dl_mcs'].replace({0:np.nan},inplace=True)
#  **** --------------------------------------------------------------------------------------------- ****
# ****************

# We are going to plot some data of interest. We care about the timestamp (horizontal axis) and the tx_brate downlink (vertical axis).
# We are going to plot the aforementioned data for 10 users (num_ues=10).
# For this purpose we are going to use data from this file: 1010123456008_metrics.csv. This is done just for illustration purposes,
# and has nothing to do with the preparation of datasets for our Random Forest (RF) Regression Algorithm.
# If we plot the data from all the combined csv files, the produced graphs will be meaningless, since each .csv file contains data
# for different instances of our network (?? is that true ??), where users exhibit different behavior.

# The following line reads the dataset for the 010123456008_metrics.csv file.
os.chdir("/Users/maria/Documents/GitHub/oran_efforts/slices_bs1")
features_metric008=pd.read_csv("1010123456008_metrics.csv") 
# We remove the emptly columns, which contain no data (i.e. NaN entries). This is important in case we want to use this dataset
# as input to the Random Forest (RF) algorithm. Without the following line the algorithm will complain.
features_metric008 = features_metric008.loc[:, ~features_metric008.columns.str.contains('^Unnamed')]
# x1000 scaling of the input to improve our algorithm
features=1000*features
#Uncomment the following line for debugging purposes
#print(features)

# The following lines return a list with the number of users
total_number_of_users = features_metric008['num_ues'].unique()
# -- Ucomment the following line for debugging purposes to print a list with the total number of users for which we have info on our dataset --
# print(total_number_of_users) 

# We only care about plotting the data of 10 users (num_ues=10). Remember: we are using the metrics file #008.
number_of_users=10

# Extrating all the existing data about 10 users (like throughput for uplink/downlink etc.)
all_data = features_metric008['num_ues'] == number_of_users
# -- Uncomment the following line for debugging purposes to see if everything works correctly --
#print(all_data.head()) 

# Putting all the rows for 10 users in a variable
final_data = features_metric008[all_data]
# -- Uncomment the following lines for debugging purposes to see if everything works correctly --
#print(final_data.head()) 
#print(final_data) 

#--------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------- Plotting! ----------------------------------------------------------------------
#  Plotting timestamp vs. throughput for 10 users from the 1010123456008_metrics.csv file.
final_data.plot('Timestamp','tx_brate downlink [Mbps]',color="red")
plt.title("Downlink Throughput VS. Time for 10 users for the actual data of the 1010123456008_metrics.csv file.")
##plt.figure(1)

# Extract only top 60 rows  of the dataset included in the 1010123456008_metrics.csv file to make the plot a little clearer
new_data = final_data.head(60)
#  Plotting again with 60 values
new_data.plot('Timestamp','tx_brate downlink [Mbps]',color="green")
plt.title("Downlink Throughput VS. Time for 10 users for the 60 head values of 1010123456008_metrics.csv file dataset for the actual data.")
##plt.figure(2)
##plt.show() 
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
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
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# --------------------------------------------------------------------------------------------------------------------------------------------------------
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------Random Forest Algorithm-----------------------------------------------------------------------
# n_estimators: indicates the number of decision trees used
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42,max_features='log2', n_jobs=-1)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs=-1)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#--------------------------------------------------------Evaluation of our algorithm---------------------------------------------------------------------
# MAE
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf, train_features, train_labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE with Repeated K-Fold cross validator: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
#The train accuracy: The accuracy of a model on examples it was constructed on.
print('Train Accuracy Score is:', rf.score(train_features, train_labels))
#--------------------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------Uncomment the following lines to plot the decision trees---------------------------------------------
'''
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('/Users/maria/Documents/GitHub/ml book examples/tree.png')
graph.write_png('tree.png')

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 9) #default precision was 1
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------With the following lines we can arithmetically see which features play the most significant part in the predictions---------------------
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

##plt.figure(3)
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical',fontsize=6)
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
##plt.show() 
#---------------------------------------------------------------------------------------------------------------------------------------------------------

'''
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 10, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('sum_granted_prbs'), feature_list.index('tx_pkts downlink')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions2 = rf_most_important.predict(test_important)

# Evaluation of our algorithm
print('Results after making predictions based on the two most important features:')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions2))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions2)))
'''

#--------------------save the actual timestamps from the dataset-------------------------
timestamps = test_features[:, feature_list.index('Timestamp')]
#--Uncomment the following line for debugging purposes--
#print(len(timestamps))
real_timestamps=timestamps
#----------------------------------------------------------------------------------------
# find the actual throughput downlink values for the actual timestamps
times=[str(int(Timestamp)) for Timestamp in timestamps]
true_data = pd.DataFrame(data = {'time': times, 'actual': test_labels})
#----------------------------------------------------------------------------------------

#-------------------------------timestamps for predictions-------------------------------
timestamps = test_features[:, feature_list.index('Timestamp')]
#--Uncomment the following line for debugging purposes--
#print(len(timestamps))
pred_timestamps=timestamps
#----------------------------------------------------------------------------------------
# find the predicted throughput downlink values for the predicted timestamps
test_times = [str(int(Timestamp))  for Timestamp in timestamps]
predictions_data = pd.DataFrame(data = {'time': test_times, 'prediction': predictions})
#----------------------------------------------------------------------------------------

'''
plt.figure(4)
# Plot the actual values
plt.plot(true_data['time'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['time'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.legend()
plt.xlabel('Timestamps'); plt.ylabel('Downlink Througput in [Mpbs]'); plt.title('Actual and Predicted Values')
plt.xticks(['1602351509723','1602351710723'	,'1602351959723'],fontsize=12) #rotation = '60'
plt.show()
'''
'''
# Moving Average Plot
plt.figure(5)
true_data['actual'].rolling(window =20).mean().plot(linewidth=1.5)
predictions_data['prediction'].rolling(window =20).mean().plot(linewidth=1.5)
plt.show()
'''

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
#'''
# reset path again for the saving of the excel files
os.chdir("/Users/maria/Documents/GitHub/oran_efforts/testing_RF_accuracy/results_and_plotting")

# save the actual and predicted downlink throughput
real.to_csv('real_1.csv', index = False,header=False)
pred.to_csv('pred_1.csv', index = False,header=False)

# save the actual timestamps from the datasets and timestamps from the testing set
df = pd.DataFrame(real_timestamps)
df.to_csv('real_time_1.csv', index=False,header=False)
df2 = pd.DataFrame(pred_timestamps)
df2.to_csv('pred_time_1.csv', index=False,header=False)

#print(type(labels))
#plt.figure(4)
#plt.hist(labels)
#plt.show()

df3 = pd.DataFrame(labels)
df3.to_csv('labels.csv', index=False,header=False)


'''
features=features/1000
predictions=predictions/1000
test_labels=test_labels/1000
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
'''

# Uncomment for debugging purposes
#print(real[0]/1000)
#print(test_labels[0]/1000)
#print(predictions[0]/1000)


## Random Search with Cross Validation
#print('Parameters currently in use:\n')
#pprint(rf.get_params())


# close all plots
plt.close('all')
