# Model part
import torch
from torch import nn
import res
import GoogleNet

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.net1 = nn.Sequential(nn.Conv2d(147,128,3, stride=1, padding=1),
                                  nn.MaxPool2d(3,stride=1, padding=1),
                                  nn.Dropout(0.2),
                                   nn.GELU())

        self.block = GoogleNet.GoogLeNet()



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, input_dict):

        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()

        x = self.net1(obs)


        x = self.block(x)


        action_logits = x




        action_mask = input_dict["obs"]["mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        out = action_logits + inf_mask
        return out






