import numpy as np
from sklearn.svm import SVC
from preprocessing.ParameterGenerator import ParameterGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



pg = ParameterGenerator()
x, y = pg.convert_to_np()

x = MinMaxScaler().fit_transform(x)

shuffle_index = np.random.permutation(x.shape[0])
x, y = x[shuffle_index], y[shuffle_index]

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)


svm_clf = SVC(kernel='linear', C=float('inf'))
svm_clf.fit(X_train, y_train)



print(svm_clf.score(X_test, y_test))


