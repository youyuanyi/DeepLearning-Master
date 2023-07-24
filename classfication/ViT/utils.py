import json
import os
import random
import sys

import torch.nn
from matplotlib import pyplot as plt
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    """
    划分训练集和验证集
    """
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    封装一个epoch的训练（不包括验证）
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).to(device)  # 一个epoch累积的训练loss
    acc_num = torch.zeros(1).to(device)  # 预测正确样本数
    sample_num = 0
    data_bar = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_bar):
        optimizer.zero_grad()
        images, labels = data
        sample_num += images.shape[0]  # batch数就是图片数量
        outputs = model(images.to(device))
        pred_class = torch.max(outputs, dim=1)[1]
        acc_num += torch.eq(pred_class, labels.to(device)).sum()

        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        train_loss += loss.detach()
        data_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                            train_loss.item() / (step + 1),
                                                                            acc_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()

    return train_loss.item() / (step + 1), acc_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    val_loss = torch.zeros(1).to(device)  # 训练loss
    acc_num = torch.zeros(1).to(device)  # 预测正确样本数
    sample_num = 0

    model.eval()
    val_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(val_bar):
        images, labels = data
        sample_num += images.shape[0]

        outputs = model(images.to(device))
        pred_class = torch.max(outputs, dim=1)[1]
        acc_num += torch.eq(pred_class, labels.to(device)).sum()

        loss = loss_function(outputs, labels.to(device))
        val_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               val_loss.item() / (step + 1),
                                                                               acc_num.item() / sample_num)

    return val_loss.item() / (step + 1), acc_num.item() / sample_num

if __name__ == '__main__':
    root = "../../data_set/flower_data/flower_photos"
    a, b, c, d = read_split_data(root, 0.2)
