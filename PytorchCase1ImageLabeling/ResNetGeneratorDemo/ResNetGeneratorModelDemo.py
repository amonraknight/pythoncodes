import torch
from PIL import Image
from torchvision import transforms

from ResNetGeneratorDemo.ResNetBlock import ResNetGenerator

netG = ResNetGenerator()

model_path = r'E:/development/GitRepository/dlwpt-code/data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

# print(netG.eval())

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

horseImg = Image.open('../TestImages/horse.jpg')

horseImg_t = preprocess(horseImg)
batch_t = torch.unsqueeze(horseImg_t, 0)
batch_out = netG(batch_t)
out_t = (batch_out.data.squeeze() + 1.0)/2.0
out_img = transforms.ToPILImage()(out_t)
out_img.save('../TestImages/horse_zebra.jpg')
