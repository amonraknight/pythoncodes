from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

n_labeled = 50

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

log_reg = LogisticRegression()
#only train with 50
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print(log_reg.score(X_test, y_test))
#accuracy: 0.827