import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

filter_field_1 = 'fixed acidity'
filter_field_2 = 'volatile acidity'
filter_field_3 = 'citric acid'
filter_field_4 = 'residual sugar'
filter_field_5 = 'chlorides'
filter_field_6 = 'free sulfur dioxide'
filter_field_7 = 'total sulfur dioxide'
filter_field_8 = 'density'
filter_field_9 = 'pH'
filter_field_10 = 'sulphates'
filter_field_11 = 'alcohol'

# available quality values are 3,4,5,6,7,8
target_field = 'quality'
# make them into 3 levels
# quality_range = [[3, 4], [5, 6], [7, 8]]
# quality_range = [[3, 4], [7, 8]]
quality_range = [[3], [4], [5], [6], [7], [8]]

data_universe = pd.read_csv(r'E:/development/DataSets/logistic_regression_dataset/winequality-red.csv')

X_uni = data_universe[
    [filter_field_1, filter_field_2, filter_field_3, filter_field_4, filter_field_5, filter_field_6, filter_field_7,
     filter_field_8, filter_field_9, filter_field_10, filter_field_11]].values
y_uni_q_values = data_universe[target_field].values

X_train, X_test, y_train_q_value, y_test_q_value = train_test_split(X_uni, y_uni_q_values, stratify=y_uni_q_values,
                                                                    random_state=29, test_size=0.2)

for q_values_of_level in quality_range:
    std = StandardScaler()
    lgs_reg_model = LogisticRegression(max_iter=10000)
    lgs_reg_pip = Pipeline([
        ('StandardScaler', std),
        ('Logistic Regression', lgs_reg_model)
    ])
    # y_train_value = [[0, 1][i == q_values_of_level[0] or i == q_values_of_level[1]] for i in y_train_q_value]
    # y_test_value = [[0, 1][i == q_values_of_level[0] or i == q_values_of_level[1]] for i in y_test_q_value]
    y_train_value = [i in q_values_of_level for i in y_train_q_value]
    y_test_value = [i in q_values_of_level for i in y_test_q_value]
    lgs_reg_pip.fit(X_train, y_train_value)
    print(lgs_reg_pip.score(X_test, y_test_value))
