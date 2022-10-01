import numpy as np
# If to download from the source
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from common_util.print_digits import plot_digits

output_image_path = r'E:/development/GitRepository/pythoncodes/ml_2_image_SDGClassification/outimage.png'
output_image_path_wrongpredit = r'E:/development/GitRepository/pythoncodes/ml_2_image_SDGClassification/outimage_wrong.png'

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)

print(X.shape)
print(y.shape)

X = MinMaxScaler().fit_transform(X)
# split data for trainsing and testing
# X_train, X_test, y_train, y_test=X[:7500],X[7500:], y[:7500], y[7500:]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, train_size=0.8)

print(X_train.shape)
print(X_test.shape)

# print the digits
plot_digits(X_test, 'test data', output_image_path)

# shuffling
shuffle_index = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# model
SGDClsr = SGDClassifier(random_state=42)
SGDClsr.fit(X_train, y_train)

# print(SGDClsr.score(X_test, y_test))

wrongSet = []
for idx in range(X_test.shape[0]):
    pdctRst = SGDClsr.predict([X_test[idx]])
    if y_test[idx] != pdctRst:
        wrongSet.append(idx)

print(len(wrongSet))

X_wrong = X_test[wrongSet]
plot_digits(X_wrong, 'wrong predicted', output_image_path_wrongpredit)
