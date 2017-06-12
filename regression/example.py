from functools import reduce

from data import x, y
from regression.linear import LinearRegression

linearRegression = LinearRegression()
linearRegression.fit(x, y)

print("Input: ", x)
print("Function: y = %d + (%.2f)x1 + (%.2f)x2" % tuple(linearRegression._coefficients))

print("Prediction: ", linearRegression.predict(x))