import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from read_data import AdvDataset
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model
from torch.utils.data import DataLoader
from classfication.ResNet.model import resnet34 as create_model


def test(model, loader, criterion):
    model.eval()
    acc, acc_loss = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        acc += (output.argmax(dim=1) == y).sum().item()
        acc_loss += loss.item()

    return acc / len(loader.dataset), acc_loss / len(loader.dataset)


def main(args):
    # 没有加对抗样本的数据
    ben_dataset = AdvDataset(args.data_dir)
    ben_loader = DataLoader(ben_dataset, batch_size=args.batch_size, shuffle=False)

    # 加了对抗样本的数据
    adv_dataset = AdvDataset(args.adv_dir)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)

    # model = create_model()
    # assert os.path.exists(args.weights), "file {} does not exist.".format(args.weights)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, args.classes)
    # model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    # model.to(device)

    model = get_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights))
    criterion = nn.CrossEntropyLoss()

    # 分别测试正常样本和对抗样本下的效果
    ben_acc, _ = test(model, ben_loader, criterion)
    adv_acc, _ = test(model, adv_loader, criterion)
    print('Begin acc: ', ben_acc)
    print('Adv acc: ', adv_acc)


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=8, help='data: batch size')
    parser.add_argument('--model', type=str, default='wrn16_10_cifar10', help='model')
    parser.add_argument('--weights', type=str, default='./weights/wrn16_10_cifar10_pgd20.pth', help='model: checkpoint')
    parser.add_argument('--data_dir', type=str, default='./benign', help='test: benign images')
    parser.add_argument('--adv_dir', type=str, default='./adv', help='test: adversarial images')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--classes', type=int, default=10, metavar='N',
                        help='number of class (default: 10)')
    args = parser.parse_args()

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
