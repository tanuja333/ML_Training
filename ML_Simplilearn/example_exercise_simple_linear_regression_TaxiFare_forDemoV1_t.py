# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:33:09 2019

@author: st
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(".//TaxiFareActualsData.csv")

distance=dataset.iloc[:, :-1].values
fare=dataset.iloc[:,-1].values
#train and split
from sklearn.model_selection import train_test_split
distance_points_train, distance_points_test, fare_train, fare_test = train_test_split(distance, fare, test_size = 0.33, random_state = 0)
#applying regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
regressor.fit(distance_points_train,fare_train)
#predict
fare_pred=regressor.predict(distance_points_test)
#intercept and co-eff
regressor.coef_
#array([19.52064424])
regressor.intercept_
#73.97633588644821
#SSE-sum of squared err
sse=np.sum(np.square(fare_test-fare_pred)) #247.2829938335709
#MSE- mean square err
mse=sse/len(fare_pred)#30.910374229196364
#RMSE-root mean square err
rmse=np.sqrt(mse)
#strengthening the model
ssr=np.sum(np.square(fare_pred-np.mean(fare_test)))
error_rate=sse/ssr
#r_sqaured_coeff --> how your model is stronger than your dummy model
r_sqaured_coeff= 1-error_rate #more r_squared_coeff is 1 then your model strength is more

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