# Model part
import torch
from torch import nn
import res4 as res
import at1


class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.net1 = nn.Sequential(nn.Conv2d(147,128,3, stride=1, padding=1),
                                  nn.MaxPool2d(3,stride=1, padding=1),
                                  nn.Dropout(0.2),
                                   nn.GELU())


        self.net2 = nn.Sequential(nn.Conv2d(147,128,1, stride=1, padding=0),
                                  nn.MaxPool2d(1,stride=1, padding=0),
                                  nn.Dropout(0.2),
                                   nn.GELU())

        a1 = [at1.Att(128)]+[res.ResnetBlock(1) for i in range(4)]+[nn.Conv2d(128,256,1, stride=1, padding=0)]+[res.ResnetBlock(2) for i in range(44)]

        self.block = nn.Sequential(*a1)


        self.pr = nn.Sequential(nn.Flatten(),
                      nn.Linear(256*8*9,1024),
                        nn.Linear(1024,235))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, input_dict):

        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()

        x1 = self.net1(obs)
        x2 = self.net2(obs)

        x = torch.cat([x1,x2],dim=2)


        x = self.block(x)


        action_logits = self.pr(x)




        action_mask = input_dict["obs"]["mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        out = action_logits + inf_mask
        return out






