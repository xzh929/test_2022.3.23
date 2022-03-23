from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from CNN_net import CNN_net
from torch import optim
from torch.nn.functional import one_hot
import torch

minist_train = datasets.MNIST(root="D:\data",train=True,transform=transforms.ToTensor(),download=True)
minist_test = datasets.MNIST(root="D:\data",train=False,transform=transforms.ToTensor(),download=True)

class Train:
    def __init__(self):
        self.train_loader = DataLoader(minist_train,batch_size=500,shuffle=True)
        self.test_loader = DataLoader(minist_test, batch_size=500, shuffle=True)

        self.net = CNN_net()
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        for epoch in range(1000):
            sum_loss = 0
            for i, (imgs, tags) in enumerate(self.train_loader):

                self.net.train()
                out = self.net(imgs)
                tags = one_hot(tags)
                loss = torch.mean((out-tags)**2)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()
            avg_loss = sum_loss/len(self.train_loader)
            print("train_loss:",avg_loss)

            sum_score = 0
            sum_test_loss = 0
            for i, (imgs,tags) in enumerate(self.test_loader):
                self.net.eval()
                test_out = self.net(imgs)
                tags = one_hot(tags)
                loss = torch.mean((test_out-tags)**2)
                sum_test_loss += loss.item()

                pre = torch.argmax(test_out,dim=1)
                lable = torch.argmax(tags,dim=1)
                score = torch.mean(torch.eq(pre,lable).float()).item()
                sum_score += score

            avg_test_loss = sum_test_loss / len(self.test_loader)
            avg_score = sum_score / len(self.test_loader)

            print("epoch:", epoch, "test_loss:", avg_test_loss, "score:", avg_score)



if __name__ == '__main__':
    train = Train()
    train()