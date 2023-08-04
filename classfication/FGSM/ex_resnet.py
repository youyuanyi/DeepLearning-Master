"""
训练resnet,vit,swin进行Fashion-MNIST数据集分类
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from classfication.ResNet.model import resnet34 as create_model


def load_data(args):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet需要输入224x224的图像
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 灰度图像只有一个通道，均值和标准差都为0.5
    ])

    # 下载Fashion-MNIST数据集并应用于数据预处理
    train_dataset = torchvision.datasets.FashionMNIST(root='../data_set/data', train=True, download=False,
                                                      transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='../data_set/data', train=False, download=False,
                                                     transform=transform)

    # 从训练集中划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def train_one_epoch(data_loader, model, loss_func, optimizer, device, cur_epoch):
    model.train()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_num = 0  # 记录总样本数

    train_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()

        accu_loss += loss
        predicted_train = torch.max(output.data, dim=1)[1]  # 计算最大值的索引
        total_num += labels.size(0)
        accu_num += (predicted_train == labels).sum()

        train_bar.desc = "train epoch[{}] loss:{:.3f} acc:{:.3f} ".format(cur_epoch + 1,
                                                                          accu_loss.item() / (step + 1),
                                                                          accu_num.item() / total_num)
    return accu_loss.item() / (step + 1), accu_num.item() / total_num


def evaluate_one_epoch(data_loader, model, loss_func, epoch, device):
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred_classes = torch.max(outputs.data, dim=1)[1]
            total_num += labels.size(0)

            accu_num += torch.eq(pred_classes, labels).sum()

            loss = loss_func(outputs, labels)
            accu_loss += loss

            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / total_num)

    return accu_loss.item() / (step + 1), accu_num.item() / total_num


# fgsm产生对抗样本
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0,1]范围,仅操作epsilon!=0的情况
    if epsilon != 0:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


# 测试训练的模型在测试集上的准确率
def test_acc(model, device, test_loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # 把数据和标签发送到设备
            data, target = data.to(device), target.to(device)

            output = model(data)
            predicted = torch.max(output.data, dim=1)[1]
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on test set: {:.2f}%'.format(accuracy))
    return accuracy


# 测试fgsm攻击
def test(model, device, test_loader, epsilon, loss_func):
    # 精度计数器
    correct = 0
    # 保存对抗样本
    adv_examples = []
    total_num = 0

    model.eval()
    # 这里不能加 with torch.no_grad()，不然loss.backward()会无法计算得到data的grad
    test_bar = tqdm(test_loader, file=sys.stdout)
    for step, data in enumerate(test_bar):
        # 把数据和标签发送到设备
        data, target = data
        data, target = data.to(device), target.to(device)

        # 追踪输入的梯度
        data.requires_grad = True

        output = model(data)
        init_pred = torch.max(output.data, dim=1)[1]

        # 如果初始预测是错误的，不打断攻击，继续
        # if torch.eq(init_pred, target).sum() != target.size(0):
        #     continue

        loss = loss_func(output, target)

        # 将所有现有的渐变归零，和optimizer.zero_grad()不同
        model.zero_grad()

        # 计算后向传递模型的梯度
        loss.backward()

        # 收集data_grad
        data_grad = data.grad.data

        # 产生对抗样本
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 测试攻击效果
        output = model(perturbed_data)

        final_pred = torch.max(output.data, dim=1)[1]
        total_num += target.size(0)
        correct += torch.eq(final_pred, target).sum()

        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            # 取batch中的第一个为例
            adv_examples.append((init_pred[0].item(), final_pred[0].item(), adv_ex))

        test_bar.desc = "test loss:{:.3f} acc:{:.3f} ".format(loss.item() / (step + 1),
                                                              correct / total_num)
    final_acc = correct / total_num
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, correct, total_num, final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = './weights/resNet34.pth'

    # 获取数据集
    train_loader, val_loader, test_loader = load_data(args)
    if args.train:
        print('Train mode')

        net = create_model()
        assert os.path.exists(args.weights), "file {} does not exist.".format(args.weights)
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))

        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, args.classes)
        net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(train_loader, net, loss_function, optimizer, device, epoch)
            val_loss, val_acc = evaluate_one_epoch(val_loader, net, loss_function, epoch, device)

            if val_acc > best_acc:
                torch.save(net.state_dict(), save_path)
                best_acc = val_acc
    else:
        print("Test mode")
        epsilons = [0, .05, .1, .15, .2, .25, .3]
        accuracies = []
        examples = []

        net = create_model()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, args.classes)
        net.load_state_dict(torch.load(save_path, map_location='cpu'))
        net.to(device)

        # 测试训练好的模型的准确率
        _ = test_acc(net, device, test_loader)

        loss_function = nn.CrossEntropyLoss()
        for eps in epsilons:
            acc, ex = test(net, device, test_loader, eps, loss_function)
            accuracies.append(acc)
            examples.append(ex)

        # 可视化
        accuracies = [acc.cpu().numpy() for acc in accuracies]
        plt.figure(figsize=(5, 5))
        plt.plot(epsilons, accuracies, '*-')
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy on different Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()

        # 在每个epsilon上绘制几个对抗样本的例子
        cnt = 0

        plt.figure(figsize=(8, 10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons), len(examples[0]), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig, adv, ex = examples[i][j]
                ex = ex[0]  # 从批次中取出第一张图

                ex_grey = np.mean(ex, axis=0)  # 计算3通道的均值，转为1通道

                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex_grey, cmap='gray')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Example')
    parser.add_argument('--train', type=bool, default=False, help='train mode or test mode')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weights', type=str, default='../ResNet/resnet34.pth',
                        help='initial weights path')
    parser.add_argument('--classes', type=int, default=10, metavar='N',
                        help='number of class (default: 10)')
    opt = parser.parse_args()

    main(opt)
