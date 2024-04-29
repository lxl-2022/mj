


import math
import torch.nn as nn
import torch



class Att(nn.Module):

    def __init__(self, inpts, b = 1, gama=2):
        super(Att, self).__init__()
        self.inputs = inpts
        self.kernel_size =  int(abs((math.log(self.inputs, 2) + b) / gama))

        if self.kernel_size % 2:
            self.kernel_size = self.kernel_size

        # 如果卷积核大小是奇数就变成偶数
        else:
            self.kernel_size = self.kernel_size + 1

        self.a1 = nn.AvgPool2d(4,9)
        self.a2= nn.Conv1d(1, 1,self.kernel_size, stride=1, padding='same')
        self.a3 = nn.Sigmoid()



    def forward(self, x):
        b_s = x.shape[0]
        n1 = x
        x = self.a1(x)
        x = x.reshape(b_s,1,self.inputs)
        x = self.a2(x)
        x = x.reshape(b_s,self.inputs,1,1)
        x = self.a3(x)
        x = torch.mul(n1,x)

        return x

