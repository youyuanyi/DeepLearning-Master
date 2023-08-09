import sys

import torch
from tqdm import tqdm


def pgd_attack(model, x, y, criterion, epsilon, alpha, num_steps=20):
    """
    初始化对抗样本 adv_x = x
    对于 step = 1 到 num_steps：
        将 adv_x 放入模型进行前向传播，得到预测结果 logits
        计算损失 loss = criterion(logits, y_target)
        计算损失关于 adv_x 的梯度 delta_grad = gradient(loss, adv_x)

        使用梯度上升更新扰动 delta：
        delta = delta + alpha * sign(delta_grad)

        对扰动进行投影，确保其范围在 [-epsilon, epsilon] 内：
        delta = clip(delta, -epsilon, epsilon)

        更新对抗样本：
        adv_x = clip(x + delta, 0, 1)

    输出：对抗样本 adv_x
    # 该函数只产生最终扰动，更新样本放到外面
    :param model:
    :param x: 输入
    :param y: 目标标签
    :param epsilon: 扰动大小
    :param alpha: 梯度上升步长
    :param num_steps: 迭代次数
    :return:
    """
    delta = torch.zeros_like(x, requires_grad=True)  # 扰动，一开始为0，即adv_x = x
    # 在进行 PGD 攻击时，我们通过多次迭代来调整扰动delta，以最大化损失函数。
    for i in range(num_steps):
        output = model(x + delta)
        loss = criterion(output, y)
        loss.backward()

        # 使用梯度上升更改delta
        d = (delta + alpha * delta.grad.detach().sign())
        # 对扰动进行投影，确保其在 [-epsilon, epsilon]内
        delta.data = d.clamp(-epsilon, epsilon)
        # 确保每次迭代都是在当前的 delta 下计算的梯度，而不会受到之前迭代中的梯度影响。这是为了保证梯度上升更新扰动的方向始终是朝着最大化损失的方向
        delta.grad.zero_()

    # 该函数只产生最终扰动，更新样本放到外面
    return delta.detach()


def train_one_epoch(args, data_loader, model, loss_func, optimizer, device, cur_epoch):
    cifar_10_mean = (0.491, 0.482, 0.447)
    cifar_10_std = (0.202, 0.199, 0.201)
    mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
    std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

    model.train()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_num = 0  # 记录总样本数

    train_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # pgd攻击找到扰动
        # delta = pgd_attack(model, inputs, labels, loss_func, args.eps / 255 / std, args.alpha / 255 / std,
        #                           args.num_steps)
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


def evaluate_one_epoch(args, data_loader, model, loss_func, device, cur_epoch):
    cifar_10_mean = (0.491, 0.482, 0.447)
    cifar_10_std = (0.202, 0.199, 0.201)
    mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
    std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    total_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # pgd攻击找到扰动
        # delta = pgd_attack(model, inputs, labels, loss_func, args.eps / 255 / std, args.alpha / 255 / std,
        #                  args.num_steps)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        pred_classes = torch.max(outputs.data, dim=1)[1]
        total_num += labels.size(0)

        accu_num += torch.eq(pred_classes, labels).sum()

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(cur_epoch + 1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / total_num)

    return accu_loss.item() / (step + 1), accu_num.item() / total_num


def epoch_adversarial(args, model, loader, criterion, device, cur_epoch, optimizer=None):
    cifar_10_mean = (0.491, 0.482, 0.447)
    cifar_10_std = (0.202, 0.199, 0.201)
    mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
    std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

    if optimizer:
        model.train()
    else:
        model.eval()

    train_acc, train_loss, total_num = 0.0, 0.0, 0
    data_loader = tqdm(loader, file=sys.stdout)
    for step, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        # 扰动
        delta = pgd_attack(model, x, y, criterion, args.eps / 255 / std, args.alpha / 255 / std, args.num_steps)
        yp = model(x+delta)
        loss = criterion(yp, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item()
        total_num += y.size(0)

        if optimizer:
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(cur_epoch + 1,
                                                                                   train_loss / (step + 1),
                                                                                   train_acc / total_num)
        else:
            data_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.3f}".format(cur_epoch + 1,
                                                                                 train_loss / (step + 1),
                                                                                 train_acc / total_num)
    return train_acc / total_num, train_loss / (step + 1)
