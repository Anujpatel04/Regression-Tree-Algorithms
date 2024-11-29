import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Tree Algorithms\emp_sal.csv')

# Split the data into independent 'X' and dependent 'Y' variables
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

# Fitting Decision Tree Classification to the Training set
regressor= DecisionTreeClassifier(random_state=0)
regressor.fit(X,Y)

# Predicting a new result
Y_pred = regressor.predict([[6.5]])
print(Y_pred)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()