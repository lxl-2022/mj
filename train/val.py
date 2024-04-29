import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import dataset_v
from torch.utils.data import DataLoader
from model20 import CNNModel
import torch.nn.functional as F
import torch
import os
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import copy




if __name__ == '__main__':

    # Load dataset

    batchSize = 1024
    device = 'cuda'
    # Load model
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    validateDataset = dataset_v.MahjongGBDataset(0,1)
    vloader = DataLoader(validateDataset, batch_size=batchSize, shuffle=False)

    logdir = '/home/lxl/data/model_2/'
    if os.path.exists(logdir + 'checkpoint'):
        pass
    else:
        os.mkdir(logdir + 'checkpoint')
    epoch = 1
    best_acc = 0
    resume = True  # 设置是否需要从上次的状态继续训练

    if resume:
        if os.path.isfile(logdir + 'checkpoint/model2.pkl'):
            print("----------LOAD----------")
            checkpoint = torch.load(logdir + 'checkpoint/model2.pkl')
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['acc']
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            epoch = 1  # 如果没进行训练过，初始训练epoch值为1
            best_acc = 0

    yb = [0 for i in range(236)]
    acc = [0 for i in range(236)]

    for e in range(1):
        print('Run validation:')
        correct1 = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False,
                          'obs': {'observation': d[0].cuda(),
                                  'mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)

                az = torch.eq(pred, d[2].cuda()).cpu().tolist()

                            
                pr = pred.cpu().numpy().tolist()

                for j,k in enumerate(pr):

                    yb[k] += 1
                    acc[k] += az[j]


        pre = [yb,acc]
        np.save(logdir + 'checkpoint/pre2.npy', pre)
