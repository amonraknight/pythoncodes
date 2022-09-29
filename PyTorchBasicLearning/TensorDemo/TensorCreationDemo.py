import torch

# Create a tensor of 3 in 1-dimension, filled by 1.
a = torch.ones(3)

print(a)
# Select by index.
print(a[0])

# Setting values.
a[2] = 3.0
print(a)

points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(points)

# 2-dimensional
points = torch.zeros(3, 2)
print(points.shape)
print(points)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)
print(points[0])
print(points[0, 1])

# All rows since row 2
print(points[1:])

# All rows since row 2, all columns since column 2
print(points[1:, 1:])

# All rows since row 2, column 1
print(points[1:, 0])

# Add one dimension.
print(points[None])

# Create a tensor of shape[channels, rows, columns] filled with random values
img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# Shape [batch, channels, rows, columns]
batch_t = torch.randn(2, 3, 5, 5)

# Get naive gray(an unweighted mean at channel) image.
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)

print(img_gray_naive.shape, batch_gray_naive.shape)

# Get weighted gray image.
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
print(unsqueezed_weights.shape, unsqueezed_weights)

# multiply each channel with weights
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)

# Sum up the weighted values on each channel
img_gray_weighted = img_weights.sum(-3)
batch_gray_weight = batch_weights.sum(-3)
print(img_weights.shape, batch_weights.shape, img_gray_weighted.shape, batch_gray_weight.shape)


