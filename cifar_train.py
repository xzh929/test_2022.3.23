from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MLP_net import MLP_Net
from torch import optim
from torch.nn.functional import one_hot
import torch

cifar_train = datasets.CIFAR10(root="F:\data", train=True, transform=transforms.ToTensor(),
                               download=True)  # 从datasets导入cifar训练数据集
cifar_test = datasets.CIFAR10(root="F:\data", train=False, transform=transforms.ToTensor(), download=True)
# 从datasets导入cifar测试数据集


class Train:
    def __init__(self):
        self.train_data = DataLoader(cifar_train, batch_size=500, shuffle=True)  # 加载训练集
        self.test_data = DataLoader(cifar_test, batch_size=500, shuffle=True)  # 加载测试集

        self.net = MLP_Net()  # 实例化网络
        self.opt = optim.Adam(self.net.parameters())  # 优化网络参数

    def __call__(self):
        for epoch in range(1000):
            sum_loss = 0.  # 训练的总损失
            # self.net.load_state_dict(torch.load())
            for i, (imgs, tags) in enumerate(self.train_data):  # 从加载器中读取图片数据和标签
                imgs = imgs.reshape(-1, 32 * 32 * 3)  # 将3维形状的图片数据转换为全连接可接收的2维数据
                self.net.train()  # 开启训练模式
                out = self.net.forward(imgs)  # 传入图片数据到网络
                tags = one_hot(tags)  # 将图片标签转换为one-hot编码
                loss = torch.mean((out - tags) ** 2)  # 计算均方差损失

                self.opt.zero_grad()  # 清空梯度
                loss.backward()  # 自动求导
                self.opt.step()  # 更新梯度

                sum_loss += loss.item()  # 损失累加
            avg_loss = sum_loss / len(self.train_data)  # 每次训练循环的平均损失
            print("train_loss:", avg_loss)
            # torch.save(self.net.state_dict())

            sum_score = 0.  # 测试总准确度
            test_sum_loss = 0.  # 测试总损失
            for i, (imgs, tags) in enumerate(self.test_data):  # 迭代取出测试集
                imgs = imgs.reshape(-1, 32 * 32 * 3)  # 将3维形状的图片数据转换为全连接可接收的2维数据
                self.net.eval()  # 开启测试模式
                test_out = self.net.forward(imgs)  # 测试数据传入模型
                tags = one_hot(tags)  # 将图片标签转换为one-hot编码
                test_loss = torch.mean((test_out - tags) ** 2)  # 计算测试均方差损失
                test_sum_loss += test_loss.item()  # 损失求和

                pre = torch.argmax(test_out, dim=1)  # 索引取出概率最大值
                label_tags = torch.argmax(tags, dim=1)  # 取出图片标签
                score = torch.mean(torch.eq(pre, label_tags).float()).item()  # 求准确度
                sum_score += score

            test_avg_loss = test_sum_loss / len(self.test_data)  # 平均测试损失
            test_avg_score = sum_score / len(self.test_data)  # 平均准确度

            print("epoch:", epoch, "test_loss:", test_avg_loss, "test_score:", test_avg_score)


if __name__ == '__main__':
    train = Train()
    train()
