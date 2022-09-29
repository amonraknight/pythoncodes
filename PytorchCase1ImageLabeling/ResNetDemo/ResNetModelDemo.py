import torch
from torchvision import models, transforms
from PIL import Image

resNet = models.resnet101(pretrained=True)

# print(resNet)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # The input data (RGB) must be standardized as the training data.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dogImage = Image.open('../TestImages/cat01.jpg')
#dogImage.show()
dogImagePd = preprocess(dogImage)


batch = torch.unsqueeze(dogImagePd, 0)

resNet.eval()
out = resNet(batch)
# This will print the vector of 1000.
# print(out)

with open('../TestImages/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0]*100
print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
for eachIdx in indices[0][:5]:
    print(labels[eachIdx], percentage[eachIdx].item())

