import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 加载预训练的ResNet模型，注意这里是ResNet18而不是ResNet50，因为MNIST数据集较小
model = torchvision.models.resnet18(pretrained=True)

# 替换ResNet的最后一层全连接层，使其适应10个MNIST类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 将模型移动到GPU（如果有可用的GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18需要输入224x224的图像
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 灰度图像只有一个通道，均值和标准差都为0.5
])

# 下载并加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练ResNet模型，这里为了简化示例，只训练1个epoch
def train_model(model, trainloader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / (i + 1)})
                pbar.update(1)

    print('Finished Training')


# 对抗样本生成函数，使用FGSM算法
def fgsm_attack(model, data, target, epsilon):
    # 对输入数据data进行梯度跟踪
    data.requires_grad = True
    # 先将干净的data输入网络，产生梯度
    outputs = model(data)
    loss = criterion(outputs, target)
    # 清空模型中所有可学系参数的梯度信息
    model.zero_grad()
    # 计算得到data的梯度
    loss.backward()

    # FGSM攻击：向原始图像添加扰动
    data_grad = data.grad.data
    # 生成对抗样本
    perturbed_data = data + epsilon * torch.sign(data_grad)
    return perturbed_data


if __name__ == "__main__":
    print("using device: ", device)

    # 训练ResNet模型，这里为了简化示例，只训练1个epoch
    train_model(model, trainloader, criterion, optimizer, num_epochs=1)

    # FGSM攻击
    epsilon = [0., 0.05, 0.1, 0.2, 0.3]  # 扰动大小
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    model.eval()

    correct = 0
    total = 0
    for ep in epsilon:
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU

            # 对抗样本生成
            perturbed_images = fgsm_attack(model, images, labels, ep)

            # 对抗样本分类
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epsilon:{ep} Accuracy on adversarial examples: {100 * correct / total:.2f}%')
