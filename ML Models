### Linear Regression 
'''
    Fundamental supervised learning algorithm used for predicting a continuous outcome. Establishes a linear relationship between the input features and the target variable. 
'''

# Linear regression is a fundamental supervised learning algorithm used for predicting a continuous outcome. 
#It establishes a linear relationship between the input features and the target variable.

# import libraries 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data 
np.random.seed(42)
x = 2*np.random.randn(100,1)
y = 4 + 3 * x + np.random.randn(100,1)

#Split the data into training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the linear regression model 
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

#make predictions on the test set 
y_pred = lin_reg.predict(x_test)

#Plot the results 
plt.scatter(x_test, y_test, color='black', label='Actual data')
plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Linear Regression model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.savefig('linear_regression_plot.png')
plt.show()
