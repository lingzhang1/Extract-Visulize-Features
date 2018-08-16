import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print(x.size())
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # print(x.size())
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


img = cv2.imread('examp_img.png')
img = cv2.resize(img,(32,32))
img = img[:,:,0]
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=0)


target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()


optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

net_output = net(torch.Tensor(img))
print(net_output)

loss = criterion(net_output, target)
print(loss)

loss.backward()
optimizer.step()
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())
