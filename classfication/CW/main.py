import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from util import CWAttack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
])

class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


def image_folder_custom_label(root, transform, custom_label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

normal_data = image_folder_custom_label(root = './data/imagenet', transform = transform, custom_label = idx2label)
normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)

normal_iter = iter(normal_loader)
images, labels = next(normal_iter)

print("True Image & True Label")
imshow(torchvision.utils.make_grid(images, normalize=True), [normal_data.classes[i] for i in labels])

print("True Image & Predicted Label")
model = models.inception_v3(pretrained=True).to(device)

model.eval()

correct = 0
total = 0

for images, labels in normal_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

    _, pre = torch.max(outputs.data, 1)

    total += 1
    correct += (pre == labels).sum()

    imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])

print('Accuracy of test text: %f %%' % (100 * float(correct) / total))

print("Attack Image & Predicted Label")

model.eval()

correct = 0
total = 0

for images, labels in normal_loader:
    cw = CWAttack(device, model, labels, targeted=False, c=0.1)
    adv_images = cw.attack(images)
    labels = labels.to(device)
    outputs = model(adv_images)

    _, pre = torch.max(outputs.data, 1)

    total += 1
    correct += (pre == labels).sum()

    imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])

print('Accuracy of test text: %f %%' % (100 * float(correct) / total))
