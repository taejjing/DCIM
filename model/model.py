import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DCIMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)

        self.out = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.dropout(self.fc1(x), p=0.5)
        x = F.dropout(self.fc2(x), p=0.5)

        x = self.out(x)
        return x


import torch

net = DCIMModel()
print(net)
sample = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32)

net.train()
y = net(sample)


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
