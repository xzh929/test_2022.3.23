from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from CNN_CifarNet import CNN_net
from ResNet50 import ResNet50
import torch
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import os

transforms_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transforms_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

cifar_train = datasets.CIFAR10(root="F:\data", train=True, transform=transforms_train, download=True)
cifar_test = datasets.CIFAR10(root="F:\data", train=False, transform=transforms_test, download=False)

Score = 0.


class Train:
    def __init__(self):
        self.train_loader = DataLoader(cifar_train, batch_size=500, shuffle=True)
        self.test_loader = DataLoader(cifar_test, batch_size=100, shuffle=True)

        self.summary = SummaryWriter("logs")

        # pytorch中的交叉熵自带对标签的one-hot处理
        self.loss_fun = nn.CrossEntropyLoss()

        # self.net = CNN_net().cuda()
        self.net = ResNet50().cuda()
        # self.opt = optim.Adam(self.net.parameters())
        self.opt = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    def __call__(self):
        global Score
        for epoch in range(1000):
            sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.train_loader):
                self.net.train()
                imgs = imgs.cuda()
                tags = tags.cuda()
                out = self.net(imgs)
                # print(out.shape)
                # print(tags.shape)
                # tags = one_hot(tags, 10)
                # loss = torch.mean((out - tags) ** 2)
                loss = self.loss_fun(out, tags)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()
            avg_loss = sum_loss / len(self.train_loader)
            print("train_loss:", avg_loss)

            sum_test_loss = 0.
            sum_score = 0.
            for i, (imgs, tags) in enumerate(self.test_loader):
                self.net.eval()
                imgs = imgs.cuda()
                tags = tags.cuda()
                out = self.net(imgs)
                # tags = one_hot(tags, 10)
                # loss = torch.mean((out - tags) ** 2).item()
                loss = self.loss_fun(out, tags)

                pre = torch.argmax(out, dim=1)
                # label = torch.argmax(tags, dim=1)
                # score = torch.mean(torch.eq(pre, label).float())
                score = torch.mean(torch.eq(pre, tags).float())
                sum_score += score.item()
                sum_test_loss += loss.item()

            avg_score = sum_score / len(self.test_loader)
            avg_test_loss = sum_test_loss / len(self.test_loader)
            self.summary.add_scalars("loss", {"train_loss": avg_loss, "test_loss": avg_test_loss}, epoch)
            self.summary.add_scalar("score", avg_score, epoch)
            print("epoch:", epoch, "test_loss:", avg_test_loss, "score:", avg_score)

            if avg_score > Score:
                Score = avg_score
                if not os.path.exists('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(self.net.state_dict(), 'checkpoint/resnet50.pth')


if __name__ == '__main__':
    train = Train()
    train()
