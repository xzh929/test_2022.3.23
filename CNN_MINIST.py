from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from CNN_net import CNN_net
from torch import optim
from torch.nn.functional import one_hot
import torch

minist_train = datasets.MNIST(root="F:\data", train=True, transform=transforms.ToTensor(),
                              download=True)  # 从datasets导入mnist训练数据集
minist_test = datasets.MNIST(root="F:\data", train=False, transform=transforms.ToTensor(),
                             download=False)  # 从datasets导入mnist测试数据集

DEVICE = "cuda"


class Train:
    def __init__(self):
        self.train_loader = DataLoader(minist_train, batch_size=250, shuffle=True)  # 加载训练集
        self.test_loader = DataLoader(minist_test, batch_size=100, shuffle=True)  # 加载测试集

        self.net = CNN_net().to(DEVICE)  # 将模型放到cuda
        self.opt = optim.Adam(self.net.parameters())  # 优化网络参数

    def __call__(self):
        for epoch in range(1000):
            sum_loss = 0.  # 训练的总损失
            for i, (imgs, tags) in enumerate(self.train_loader):
                self.net.train()
                imgs = imgs.to(DEVICE)  # 将图片和标签数据放到cuda，先执行该操作再将数据传入网络
                tags = tags.to(DEVICE)
                out = self.net(imgs)  # 测试数据传入网络得到输出
                tags = one_hot(tags, 10)  # 标签转换为one-hot编码
                loss = torch.mean((out - tags) ** 2)  # 训练集均方差损失

                self.opt.zero_grad()  # 清空梯度
                loss.backward()  # 自动求导
                self.opt.step()  # 更新梯度

                sum_loss += loss.item()  # 训练总损失
            avg_loss = sum_loss / len(self.train_loader)  # 训练平均损失，总损失除以数据加载长度，即训练循环次数
            print("train_loss:", avg_loss)

            sum_score = 0.  # 测试总准确率
            sum_test_loss = 0.  # 测试总损失
            for i, (imgs, tags) in enumerate(self.test_loader):
                self.net.eval()
                imgs = imgs.to(DEVICE)  # 将图片和标签数据放到cuda，先执行该操作再将数据传入网络
                tags = tags.to(DEVICE)
                test_out = self.net(imgs)
                tags = one_hot(tags, 10)
                loss = torch.mean((test_out - tags) ** 2)  # 测试的均方差损失
                sum_test_loss += loss.item()  # 每次损失求和得到测试总损失

                pre = torch.argmax(test_out, dim=1)  # 将模型的输出从one-hot反编码，求输出值中的最大值
                label = torch.argmax(tags, dim=1)  # 将标签从one-hot反编码
                score = torch.mean(torch.eq(pre, label).float()).item()  # 比较模型预测与实际标签的正误，求得平均值得到模型的测试准确度
                sum_score += score  # 模型预测总准确度

            avg_test_loss = sum_test_loss / len(self.test_loader)  # 测试平均损失
            avg_score = sum_score / len(self.test_loader)  # 测试平均准确度

            print("epoch:", epoch, "test_loss:", avg_test_loss, "score:", avg_score)


if __name__ == '__main__':
    train = Train()
    train()
