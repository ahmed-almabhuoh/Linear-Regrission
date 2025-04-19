import numpy as np
from sklearn.linear_model import LinearRegression

# Data from the table
Y = np.array([43, 63, 71, 61, 81, 43, 58, 71, 72, 67, 64, 67, 69, 68, 77, 81, 74, 65, 65, 50, 50, 64, 53, 40, 63, 66, 78, 48, 85, 82])
X1 = np.array([51, 64, 70, 63, 78, 55, 67, 75, 82, 61, 53, 60, 62, 83, 77, 90, 85, 60, 70, 58, 40, 61, 66, 37, 54, 77, 75, 57, 85, 82])
X2 = np.array([30, 51, 68, 45, 56, 49, 42, 50, 72, 45, 53, 47, 57, 83, 54, 50, 64, 65, 46, 68, 33, 52, 52, 42, 42, 66, 58, 44, 71, 39])
X3 = np.array([39, 54, 69, 47, 66, 44, 56, 55, 67, 47, 58, 39, 42, 45, 72, 72, 69, 75, 57, 54, 34, 62, 50, 58, 48, 63, 74, 45, 71, 59])
X4 = np.array([61, 63, 76, 54, 71, 54, 66, 70, 71, 62, 58, 59, 55, 59, 79, 60, 79, 55, 75, 64, 43, 66, 63, 50, 66, 88, 80, 51, 77, 64])
X5 = np.array([92, 73, 86, 84, 83, 49, 68, 66, 83, 80, 77, 74, 63, 77, 77, 54, 79, 80, 85, 78, 64, 80, 80, 57, 75, 76, 78, 83, 74, 78])
X6 = np.array([45, 47, 48, 35, 47, 34, 35, 41, 31, 41, 34, 41, 25, 35, 46, 36, 63, 60, 46, 52, 33, 41, 37, 49, 33, 72, 49, 38, 55, 39])

# Combine all X variables into a single 2D array
X = np.column_stack((X1, X2, X3, X4, X5, X6))

# Create and fit the multiple linear regression model
model = LinearRegression()
model.fit(X, Y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the results
print(f"Multiple Linear Model: Y = {intercept:.2f} + {coefficients[0]:.2f}*X1 + {coefficients[1]:.2f}*X2 + {coefficients[2]:.2f}*X3 + {coefficients[3]:.2f}*X4 + {coefficients[4]:.2f}*X5 + {coefficients[5]:.2f}*X6")
print(f"Intercept: {intercept:.2f}")
print(f"Coefficients: X1={coefficients[0]:.2f}, X2={coefficients[1]:.2f}, X3={coefficients[2]:.2f}, X4={coefficients[3]:.2f}, X5={coefficients[4]:.2f}, X6={coefficients[5]:.2f}")