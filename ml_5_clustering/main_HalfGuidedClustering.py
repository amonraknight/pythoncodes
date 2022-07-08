import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

print(X_digits.shape)

# The first round of clustering, to get the representatives.
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)

# Find the most representative according to the distance to each center.
representative_digits_idx = np.argmin(X_digits_dist, axis=0)
X_representatives = X_train[representative_digits_idx]
y_representatives = y_train[representative_digits_idx]

# drawDigits(X_representatives, k)

# Use the representative in regression.
log_reg = LogisticRegression()
# only train with 50
log_reg.fit(X_representatives, y_representatives)
print(log_reg.score(X_test, y_test))
# 0.92

# Tag all the train data.
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representatives[i]

log_reg2 = LogisticRegression()
log_reg2.fit(X_train, y_train_propagated)
print(log_reg2.score(X_test, y_test))
# 0.928


percentile_closest = 20
# Find the distance of each item to the closest cluster center.
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
# Get the 1st 20.
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    # get the first 20% closest to each center
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg3 = LogisticRegression(random_state=42)
log_reg3.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg3.score(X_test, y_test))