from torchvision import datasets
from torchvision import transforms
import src.common.config as cfg


def get_cifar10():
    data_path = cfg.CIFAR10_PATH
    cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4813, 0.4468), (0.2470, 0.2435, 0.2616))]))
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4813, 0.4468), (0.2470, 0.2435, 0.2616))]))
    return cifar10, cifar10_val


def get_cifar2():
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']
    cifar10, cifar10_val = get_cifar10()
    cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
    cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]
    return cifar2, cifar2_val
