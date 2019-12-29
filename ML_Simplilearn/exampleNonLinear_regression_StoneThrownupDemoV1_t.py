# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:55:29 2019

@author: st
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
dataset=pd.read_csv(".//StonethrownupData.csv")
time= dataset.iloc[:, 0:1].values
distance = dataset.iloc[:, 2].values
plt.scatter(time, distance, color="orange")

# Lets try Linear Regression to the Stone thrown up dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(time, distance)

# The average or bad results
# With linear Regression
plt.scatter(time, distance, color = 'green')
plt.plot(time, lin_reg.predict(time), color = 'brown')
plt.title('Linear Regression gives erroneous results here')
plt.xlabel('Time')
plt.ylabel('height(or Dsiplacement)')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
time_poly = poly_reg.fit_transform(time)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(time_poly, distance)
pred2 = lin_reg_2.predict(poly_reg.fit_transform(time))
print("R2 coefficient for degree 4: " , r2_score(pred2,distance))


poly_reg = PolynomialFeatures(degree = 12)
time_poly = poly_reg.fit_transform(time)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(time_poly, distance)
pred3 = lin_reg_2.predict(poly_reg.fit_transform(time))
print("R2 coefficient for degree 12: " , r2_score(pred3,distance))