import glob
import os

import torch
from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
])


def GetTrainLoader(batch_size):
    train_set = CIFAR10(root='../data_set', train=True, download=False, transform=transform)
    # 从训练集中划分训练集和验证集
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def GetTestLoader(batch_size):
    test_set = CIFAR10(root='../data_set', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


class AdvDataset(Dataset):
    def __init__(self, data_dir, transform=transform):
        self.images = []  # 图像文件路径
        self.labels = []  # 标签
        self.names = []   # 图像文件名
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            # 获取当前类别目录下所有图像文件的路径
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform

    def __getitem__(self, item):
        image = self.transform(Image.open(self.images[item]))
        label = self.labels[item]
        return image, label

    def __getname__(self):
        return self.names

    def __len__(self):
        return len(self.images)
