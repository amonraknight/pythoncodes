import matplotlib.pyplot as plt
import numpy as np

import testdata.PolynomialDataGenerator as SampleDataGenerator

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X, y = SampleDataGenerator.get_2_degree_data()

mesh_size = 0.1

# polynomial conversion
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_ploy = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_ploy, y)

# prepare the model result data

x_min, x_max = X.min(), X.max()
x_range = np.arange(x_min, x_max, mesh_size)
x_range = x_range.reshape(-1, 1)
x_ploy_range = poly_features.transform(x_range)
pred_result = lin_reg.predict(x_ploy_range)

# draw
plt.plot(X, y, 'b.')

plt.plot(x_range, pred_result, 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.axis([-4, 4, -4, 11])
plt.show()
