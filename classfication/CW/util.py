import sys

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
class CWAttack:
    def __init__(self, device, model, labels, targeted=False, c=1e-4, kappa=0, max_iterations=1000):
        """
           CW攻击类的初始化函数。

           参数:
               model (nn.Module): 目标神经网络模型，用于生成对抗样本。
               targeted (bool): 是否进行有目标攻击，默认为False。
               c (float): 用于平衡目标类别和对抗样本相似性的系数，默认为1e-4。
               kappa (float): 控制对抗样本与原始样本的相似性的参数，默认为0。
               max_iterations (int): CW攻击的最大迭代次数，默认为1000。
       """
        self.device = device
        self.model = model
        self.labels = labels
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.max_iterations = max_iterations

    def f(self, x):
        outputs = self.model(x)
        # 用torch.eye创建一个独热编码矩阵，用于将类别标签转为独热向量，每一行对应一个类别标签，只有对应类别的位置上的元素为1，其余位置上的元素为0
        one_hot_labels = torch.eye(len(outputs[0]),device=self.device)[self.labels]

        # 计算目标类别的输出值和非目标类别的最大输出值
        target_output, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        other_output = torch.masked_select(outputs, one_hot_labels.byte())

        """
          在 CW 攻击中，我们尝试最小化 target_output - other_output，以实现对抗性样本的生成。
          对于有目标攻击，我们希望 target_output 足够大，other_output 足够小，以便将预测导向目标类别。
          而对于无目标攻击，我们希望 other_output 足够大，target_output 足够小，以便将预测从目标类别移开
        """

        # 如果是有目标攻击，则返回对抗性差异（使其他类别概率最大的差异）
        if self.targeted:
            return torch.clamp(target_output - other_output, min=-self.kappa)

        # 如果是无目标攻击，则返回对抗性差异（使目标类别概率最大的差异）
        else:
            return torch.clamp(other_output - target_output, min=-self.kappa)

    def attack(self, input_image):
        # 采用论文中盒约束使用的换元法，对w进行优化而不是对δ进行优化
        w = torch.tensor(torch.zeros_like(input_image, requires_grad=True)).to(self.device)
        # cw攻击是基于优化的攻击，作者实验时发现Adam收敛最快，能很好地配合换元法
        # 用Adam对w不断进行优化
        optimizer = optim.Adam([w], lr=0.01)
        prev = 1e10

        for step in range(self.max_iterations):
            # 生成对抗图像
            a = 1 / 2 * (nn.Tanh()(w) + 1).to(self.device)

            # 目标函数: minimize ||δ||p+c*f(x+δ)
            #         such that x+δ ∈ [0,1]^n
            # 先计算原始图像和对抗样本图像之间的差距:L2范数的平方
            l2_norm = nn.MSELoss(reduction='sum')(a, input_image.to(self.device))
            # 然后计算c*f(x+δ)：将对抗性差异f(a)的每个元素乘以元素c，然后求和得到一个标量值
            loss2 = torch.sum(self.c*self.f(a))
            loss = l2_norm + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.max_iterations // 10) == 0:
                if loss > prev:
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = loss

            print('- Learning Progress : %2.2f %%        ' % ((step + 1) / self.max_iterations * 100), end='\r')
        # 得到最终优化好后的w后生成最终的对抗图像
        attack_images = 1/2*(nn.Tanh()(w) + 1)

        return attack_images

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
        # 对抗图像
        cw = CWAttack(device, model, y, targeted=False, c=0.1)
        adv_images = cw.attack(x)
        yp = model(adv_images)
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
