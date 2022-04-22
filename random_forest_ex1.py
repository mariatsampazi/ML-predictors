# Python code used as example for data manipulation.
# The csv file used for this example can be found in the following path: https://github.com/wineslab/colosseum-oran-commag-dataset/blob/main/slice_mixed/rome_slow_close/tr0/exp1/bs1/slices_bs1/1010123456008_metrics.csv
# The base for the code can be found here: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from sqlite3 import Timestamp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# read the data (the data is in tidy data format)
features=pd.read_csv('1010123456008_metrics.csv') # actual data
features = features.loc[:, ~features.columns.str.contains('^Unnamed')]

# replace 0 values with NaN, used later for plotting! Not needed for our case.
#features.replace(0, np.nan,inplace=True) 
#features['dl_mcs'] = features['dl_mcs'].replace({0:np.nan},inplace=True)

#-----------------------------------------------------------------------------------------------------------------------
# uncomment the following lines to check if the .csv file is read correctly. The check is done with the first 5 entries
#print(features.head(5))
#print('The shape of our features is:', features.shape)
#print(features.describe()) # Descriptive statistics for each column
#-----------------------------------------------------------------------------------------------------------------------

# We are going to plot some data of interest. We care about the timestamp (horizontal axis) and the tx_brate downlink (vertical axis).
# We are going to plot the aforementioned data for 10 users (num_ues=10).

# The following lines return a list with the number of users
total_number_of_users = features['num_ues'].unique()
#print(total_number_of_users) #debugging

# We only care about plotting the data of 10 users (num_ues=10)
number_of_users=10

# Extrating all the existing data about 10 users
all_data = features['num_ues'] == number_of_users
#print(all_data.head()) #for debugging purposes to see if everything works correctly

# Putting all the rows for 10 users in a variable
final_data = features[all_data]
#print(final_data.head()) #debugging
#print(final_data) #debugging

#   ----- Plotting! -----

#  Plotting timestamp vs. throughput
final_data.plot('Timestamp','tx_brate downlink [Mbps]',color="red")
plt.title("Downlink Throughput VS. Time for 10 users for the actual data")
plt.figure(1)

# Extract only top 60 rows to make the plot a little clearer
new_data = final_data.head(60)
#  Plotting again with 60 values
new_data.plot('Timestamp','tx_brate downlink [Mbps]',color="green")
plt.title("Downlink Throughput VS. Time for 10 users for the 60 head values of the dataset for the actual data")
plt.figure(2)

#plt.show() #plotting

# Labels are the values we want to predict, we want to predict the downlink throughput
labels=np.array(features['tx_brate downlink [Mbps]'])
#labels = np.nan_to_num(labels) # handle emptry entries, otherwise the algorithm will complain (!!!)

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('tx_brate downlink [Mbps]', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
#print(feature_list)

# Convert to numpy array
features = np.array(features)
# handle empty entries, otherwise the algorithm will complain (!!!)
#features= np.nan_to_num(features)
#print(features) #debugging

#       ----- Training & Testing Sets -----

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# state is set to 42 for reproducible results i.e.
# the results will be the same each time I run the split for reproducible results
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#   ----- uncomment the following lines for debugging purposes -----
'''
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
'''
# ------------------------------------------------------------------
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)
'''
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42,bootstrap=False)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

from sklearn import metrics

# Evaluation of our algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
#mse = metrics.mean_squared_error(test_labels, predictions)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
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

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
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

import matplotlib.pyplot as plt
plt.figure(3)

plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical',fontsize=6)
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
#plt.show() #plotting

# Use datetime for creating date objects for plotting
import datetime

timestamps = features[:, feature_list.index('Timestamp')]
#print(len(timestamps))

times=[str(int(Timestamp)) for Timestamp in timestamps]
#print(times)

true_data = pd.DataFrame(data = {'time': times, 'actual': labels})
#print(true_data)

# timestamps for predictions
timestamps = test_features[:, feature_list.index('Timestamp')]
#print(len(timestamps))

# Column of dates
test_times = [str(int(Timestamp))  for Timestamp in timestamps]
#print(test_times)

# Dataframe with predictions and timestamps
predictions_data = pd.DataFrame(data = {'time': test_times, 'prediction': predictions})
#print(predictions_data)

plt.figure(4)
# Plot the actual values
plt.plot(true_data['time'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['time'], predictions_data['prediction'], 'ro', label = 'prediction')

plt.legend()
plt.xlabel('Timestamps'); plt.ylabel('Downlink Througput in [Mpbs]'); plt.title('Actual and Predicted Values')
plt.xticks(['1602351509723','1602351710723'	,'1602351959723'],fontsize=12) #rotation = '60'
plt.show()

# close all plots
plt.close('all') #plotting

