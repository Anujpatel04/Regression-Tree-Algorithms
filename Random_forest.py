import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Tree Algorithms\emp_sal.csv')

X= data.iloc[:, 1:2].values
Y= data.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=15, random_state=0,min_samples_split=6)
regressor.fit(X,Y)

# Predicting a new result
Y_pred = regressor.predict([[6.5]])

# Visualising the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
print(Y_pred)
