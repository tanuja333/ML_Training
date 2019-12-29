# Example for Polynomial Regression
"""
Created on Wed May  9 06:59:15 2018

@author: Shivendra
This is Non Linear Model illustration code for ML training
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Lets read the Free fall dataset
dataset = pd.read_csv('/Users/apple/AnacondaProjects/polynomRegression/Polynomial_Regression/StoneTrajectoryData.csv')
time = dataset.iloc[:, 0:1].values
# We need to read the column no 3 which has the distance or height data
distance = dataset.iloc[:, 2].values
plt.scatter(time, distance, color="orange")



# Lets try Linear Regression to the Stone thrown up dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(time, distance)

# Fitting Polynomial Regression to the dataset
#We use a tric. we transform our input x to x, x-square, x-cube....
#on these derived features we run the lienar regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
time_poly = poly_reg.fit_transform(time)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(time_poly, distance)
#pred2 = lin_reg_2.predict(poly_reg.fit_transform(time))
pred4 = lin_reg_2.predict(poly_reg.fit_transform(time))

poly_reg3 = PolynomialFeatures(degree = 12)
time_poly3 = poly_reg3.fit_transform(time)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(time_poly3, distance)

pred12 = lin_reg_3.predict(poly_reg3.fit_transform(time))
print("R2 coefficient for degree 4: " , r2_score(pred4,distance))
print("R2 coefficient for degree 12: " , r2_score(pred12,distance))

# The average or bad results
# With linear Regression
plt.scatter(time, distance, color = 'green')
plt.plot(time, lin_reg.predict(time), color = 'brown')
plt.title('Linear Regression gives erroneous results here')
plt.xlabel('Time')
plt.ylabel('height(or Dsiplacement)')
plt.show()

# Visualising the Polynomial Regression results, the good one
plt.scatter(time, distance, color = 'green')
plt.plot(time, lin_reg_2.predict(poly_reg.fit_transform(time)), color = 'brown')
plt.title('Seems to work well (Polynomial Regression)')
plt.xlabel('Time')
plt.ylabel('Height')
plt.show()

#for degree 12
plt.scatter(time, distance, color = 'green')
plt.plot(time, lin_reg_3.predict(poly_reg3.fit_transform(time)), color = 'brown')
plt.title('Seems to work well (Polynomial Regression)')
plt.xlabel('Time')
plt.ylabel('Height')
plt.show()


# Predicting a new result with Linear Regression
lin_reg.predict(18)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(18))
