import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from read_data import GetTrainLoader, GetTestLoader
from classfication.ResNet.model import resnet34 as create_model
from pytorchcv.model_provider import get_model
import torch
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util import epoch_adversarial
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = './weights/wrn16_10_cifar10_pgd20.pth'

    # 读取CIFAR10数据集
    train_loader, val_loader = GetTrainLoader(args.batch_size)
    test_loader = GetTestLoader(args.batch_size)

    # 创建resnet34模型
    # net = create_model()
    # assert os.path.exists(args.weights), "file {} does not exist.".format(args.weights)
    # net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    #
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, args.classes)
    # net.to(device)

    # wrn16_10_cifar10模型是一个更加深和宽的resnet
    net = get_model(args.model, pretrained=True).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    # 定义tensorboard
    tb_writer = SummaryWriter('./log')

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_acc, train_loss = epoch_adversarial(args, net, train_loader, loss_function, device, epoch, optimizer)

        # val
        val_acc, val_loss = epoch_adversarial(args, net, val_loader, loss_function, device, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

            # tensorboard记录数据
            tb_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
            tb_writer.add_scalars('Acc', {'train': train_acc, 'val': val_acc}, epoch + 1)

    tb_writer.close()

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=256, help='data: batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='train: learning rate')
    # 使用更深更宽的resnet
    parser.add_argument('--model', type=str, default='wrn16_10_cifar10', help='model')
    parser.add_argument('--epochs', type=int, default=20, help='train: epoch')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--save_dir', type=str, default='ckpt', help='checkpoint')

    parser.add_argument('--eps', type=float, default=8, help='pgd: epsilon')
    parser.add_argument('--alpha', type=float, default=0.8, help='pgd: alpha')
    parser.add_argument('--num_steps', type=int, default=20, help='pgd: num_steps')
    # 使用标准resnet34
    parser.add_argument('--weights', type=str, default='../ResNet/resnet34.pth',
                        help='initial weights path')
    parser.add_argument('--classes', type=int, default=10, metavar='N',
                        help='number of class (default: 10)')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
