import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import src.training.train_1 as train
from src.modules.net1 import Net
from src.datasets import get_datasets
import src.common.config as cfg


def train_nd_save_model():
    # Prepare the device:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    # Prepare training and valiation data:
    cifar2, cifar2_val = get_datasets.get_cifar2()

    '''
    img, _ = cifar2[0]
    plt.imshow(img)
    plt.show()
    '''

    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

    # Initiate the model:
    # model = Net()
    model = Net().to(device=device)

    # Initiate optimizer:
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    # Loss function:
    loss_fn = nn.CrossEntropyLoss()

    # train.training_loop_1(n_epochs=100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)
    train.training_loop_2(n_epochs=100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader,
                          device=device)

    torch.save(model.state_dict(), cfg.BIRD_PLANE_TRAINED_MODEL_PATH)
    return model


def read_model():
    # Prepare the device:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    loaded_model = Net().to(device=device)
    loaded_model.load_state_dict(torch.load(cfg.BIRD_PLANE_TRAINED_MODEL_PATH, map_location=device))
    return loaded_model


if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")

    modelo = Net()
    if os.path.exists(cfg.BIRD_PLANE_TRAINED_MODEL_PATH):
        modelo = read_model()
    else:
        modelo = train_nd_save_model()

    cifar2, cifar2_val = get_datasets.get_cifar2()
    train_loadero = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
    val_loadero = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=True)

    train.validate_2(modelo, train_loadero, val_loadero, device)

