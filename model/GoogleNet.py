import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=235,init_weights=False):
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=1, padding=1)

        self.inceptionA = nn.Sequential(Inception(128, 64, 64, 32, 32),Inception(128, 64, 64, 32, 32),Inception(128, 64, 64, 32, 32),Inception(128, 64, 64, 32, 32))
        self.conv2 = BasicConv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.inceptionB = nn.Sequential(Inception(256, 128, 128, 64, 64), Inception(256, 128, 128, 64, 64), Inception(256, 128, 128, 64, 64), Inception(256, 128, 128, 64, 64))
        self.conv3 = BasicConv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.inceptionC = nn.Sequential(Inception(512, 256, 256, 128, 128), Inception(512, 256, 256, 128, 128), Inception(512, 256, 256, 128, 128), Inception(512, 256, 256, 128, 128))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512*4*9, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.inceptionA(x)
        x = self.conv2(x)
        x = self.inceptionB(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inceptionC(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3,pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x