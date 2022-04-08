import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

filter_field_1 = 'smoker'
filter_field_2 = 'children'

input_parameter_name_1 = 'age'
input_parameter_name_2 = 'bmi'
output_parameter_name = 'charges'

mesh_size = 0.1

# read file
data_universe = pd.read_csv(r'E:/development/DataSets/linear_regression_testdata/insurance.csv')
# print(data_universe.count)
# choose the linear-like group
data_universe = data_universe[
    (data_universe[output_parameter_name] <= 15000) & (data_universe[filter_field_1] == 'no') & (
            data_universe[filter_field_2] <= 1)]
#print(data_universe.count)
# prepare the data from training and test
data_train = data_universe.sample(frac=0.9)
#data_train = data_universe.iloc[0:600]
data_test = data_universe.drop(data_train.index)

X_train = data_train[[input_parameter_name_1, input_parameter_name_2]].values
y_train = data_train[output_parameter_name].values

X_test = data_test[[input_parameter_name_1, input_parameter_name_2]].values
y_test = data_test[output_parameter_name].values


# use step regression
# scalers
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train_s = ss_X.fit_transform(X_train)
X_test_s = ss_X.transform(X_test)

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# Need to reshape the array.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train_s = ss_y.fit_transform(y_train)
y_test_s = ss_y.transform(y_test)

y_train_s = y_train_s.reshape(1, -1)[0]
y_test_s = y_test_s.reshape(1, -1)[0]

reg = SGDRegressor()
for rd in range(50):
    reg.partial_fit(X_train_s, y_train_s)
    print(reg.score(X_test_s, y_test_s))


# prepare for the chart: age/bmi range
p1_min, p1_max = data_universe[input_parameter_name_1].min(), data_universe[input_parameter_name_1].max()
p2_min, p2_max = data_universe[input_parameter_name_2].min(), data_universe[input_parameter_name_2].max()

p1_range = np.arange(p1_min, p1_max, mesh_size)
p2_range = np.arange(p2_min, p2_max, mesh_size)
p1_c, p2_c = np.meshgrid(p1_range, p2_range)

p_c = np.c_[p1_c.ravel(), p2_c.ravel()]
p_c_s = ss_X.transform(p_c)

# Run model
pred_result = reg.predict(p_c_s)
# Restore the predictions back to the original data.
pred_result = pred_result.reshape(-1, 1)
pred_result = ss_y.inverse_transform(pred_result)
pred_result = pred_result.reshape(1, -1)[0]
pred_result = pred_result.reshape(p1_c.shape)

fig = px.scatter_3d(data_universe, x=input_parameter_name_1, y=input_parameter_name_2, z=output_parameter_name)
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=p1_range, y=p2_range, z=pred_result, name='cost_surface'))
fig.show()
