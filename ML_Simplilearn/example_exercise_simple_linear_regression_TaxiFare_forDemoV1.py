# Simple Linear Regression
"""
Created on Wed May  9 06:59:15 2018

@author: Shivendra
This is Simple Linear Model illustration code for ML training
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Now, Lets import the data set


# Read from data file
# slice the input and output data

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
distance_points_train, distance_points_test, fare_train, fare_test = train_test_split(distance, fare, test_size = 0.33, random_state = 0)



# Fitting Simple Linear Regression Model to the taxi fare training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(distance_points_train, fare_train)

# Predicting the Test set results
#fare_pred = 

# Visualising the Training set results
plt.scatter(distance_points_train, fare_train, color = 'red')
plt.plot(distance_points_train, regressor.predict(distance_points_train), color = 'blue')
plt.title('Distance Vs fare (Training set)')
plt.xlabel('Dsitance')
plt.ylabel('Fare')
plt.show()

# Visualising the Test set results
plt.scatter(distance_points_test, fare_test, color = 'red')
plt.plot(distance_points_test, regressor.predict(distance_points_test), color = 'blue')
plt.title('Distance Vs Fare(Test set)')
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()