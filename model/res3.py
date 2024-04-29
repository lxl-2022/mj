import torch.nn as nn
import torch


class ResnetBlock(nn.Module):

    def __init__(self,n):
        super(ResnetBlock, self).__init__()

        self.n = n
        self.block1 =  nn.Sequential(nn.Conv2d(self.n*128,self.n*256,1, stride=1, padding=0),
                                    nn.BatchNorm2d(self.n*256),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n*256,self.n*256,3, stride=1, padding=1,groups=4),
                                    nn.BatchNorm2d(self.n*256),
                                     nn.Conv2d(self.n*256,self.n*128,1, stride=1, padding=0),
                                    nn.BatchNorm2d(self.n*128),
        )
        self.r1 = nn.ReLU()



    def forward(self, x):
        x1 = self.block1(x)

        x = self.r1(x1 + x)


        return x


