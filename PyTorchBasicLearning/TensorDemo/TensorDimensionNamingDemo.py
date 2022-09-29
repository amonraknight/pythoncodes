import torch

img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.0722])
batch_t = torch.randn(2, 3, 5, 5)

img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)
batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)
print(img_gray_weighted_fancy.shape, batch_gray_weighted_fancy.shape)

# experimental feature
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
# weights_named = torch.tensor([0.2126, 0.7152, 0.0722])

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'batches', 'channels', 'rows', 'columns')

print('img named: ', img_named.shape, img_named.names)
print('batch named: ', batch_named.shape, batch_named.names)

# weights_named must have names on all the existing dimensions.
weights_aligned = weights_named.align_as(img_named)
print('aligned named: ', weights_aligned.shape, weights_aligned.names)

# Designate a dimension where the calculation happens.
# To get the gray img, the multiplication and sum happen at color channels.
gray_named = (img_named * weights_aligned).sum('channels')
print('gray named: ', gray_named.shape, gray_named.names)

# Rename the dimensions on a tensor.
gray_plain = gray_named.rename(None)
print('gray named: ', gray_named.shape, gray_named.names)

