import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


import testdata.PolynomialDataGenerator as SampleDataGenerator

X, y, x_range = SampleDataGenerator.get_2_degree_data()

#draw the source data
plt.plot(X, y, 'k.')

for style, width, degree in (('g-', 1, 20), ('b--', 2, 2), ('r+', 3, 1)):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std = StandardScaler()
    line_reg = LinearRegression()
    poly_reg_pip = Pipeline([
        ('poly_features', poly_features),
        ('StandardScaler', std),
        ('linear_regression', line_reg)
    ])
    poly_reg_pip.fit(X, y)
    predict_result=poly_reg_pip.predict(x_range)
    plt.plot(x_range, predict_result, style, label=str(degree), linewidth=width)

plt.axis([-4, 4, -4, 11])
plt.legend()
plt.show()
