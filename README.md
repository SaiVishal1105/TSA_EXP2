# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 19-08-25
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
file_path = '/mnt/data/car_price_prediction.csv'
df = pd.read_csv(file_path)

# Select relevant columns: Production Year vs Price
data = df[['Prod. year', 'Price']].dropna()

X = data[['Prod. year']].values
y = data['Price'].values

# ----- Linear Regression -----
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# ----- Polynomial Regression (degree 2) -----
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# ----- Linear Plot -----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_linear_pred, color="black", linestyle="--", label="Linear Trend")
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("Linear Trend Estimation plot")
plt.legend()
plt.show()

# ----- Polynomial Plot -----
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
sorted_zip = sorted(zip(X.flatten(), y_poly_pred))
X_sorted, y_poly_sorted = zip(*sorted_zip)
plt.plot(X_sorted, y_poly_sorted, color="black", linestyle="--", label="Polynomial Trend (deg=2)")
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("Polynomial Trend Estimation (Degree 2) plot")
plt.legend()
plt.show()
```
### OUTPUT
<img width="1578" height="95" alt="image" src="https://github.com/user-attachments/assets/475faad9-ead7-4687-a0be-fc08d5afe1c3" />

A - LINEAR TREND ESTIMATION
<img width="1027" height="586" alt="image" src="https://github.com/user-attachments/assets/cab5157a-cc24-4838-b3f8-256b2e2a9cd5" />

B- POLYNOMIAL TREND ESTIMATION
<img width="1043" height="601" alt="image" src="https://github.com/user-attachments/assets/36ec1eda-99e0-4d45-9bc6-0de2e6f14472" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
