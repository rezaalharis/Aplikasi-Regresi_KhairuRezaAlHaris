Python 3.11.2 (v3.11.2:878ead1ac1, Feb  7 2023, 10:02:41) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

... # Data
... x = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
... y = data['Performance Index'].values
... 
... # Model regresi linear
... linear_model = LinearRegression()
... linear_model.fit(x, y)
... y_linear_pred = linear_model.predict(x)
... 
... # Model regresi eksponensial
... def exponential_model(x, a, b):
...     return a * np.exp(b * x)
... 
... popt, _ = curve_fit(exponential_model, x.flatten(), y)
... y_exponential_pred = exponential_model(x, *popt)
... 
... # Plot hasil regresi
... plt.figure(figsize=(10, 6))
... plt.scatter(x, y, color='blue', label='Data points')
... plt.plot(x, y_linear_pred, color='red', linestyle='-', label='Linear Regression')
... plt.plot(x, y_exponential_pred, color='green', linestyle='--', label='Exponential Regression')
... plt.xlabel('Jumlah Latihan Soal (NL)')
... plt.ylabel('Nilai Ujian')
... plt.title('Regresi Linear dan Eksponensial Jumlah Latihan Soal terhadap Nilai Ujian')
... plt.legend()
... plt.grid(True)
... plt.show()
... 
... # Koefisien model linear
... slope = linear_model.coef_[0]
... intercept = linear_model.intercept_
... print(f'Linear Model: y = {slope:.2f}x + {intercept:.2f}')
... 
... # Koefisien model eksponensial
... a, b = popt
... print(f'Exponential Model: y = {a:.2f} * exp({b:.2f}x)')
