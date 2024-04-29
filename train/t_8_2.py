import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import dataset
import dataset_v
from torch.utils.data import DataLoader
from model20 import CNNModel
import torch.nn.functional as F
import torch
import os
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import copy


ver = "2"


if __name__ == '__main__':

    # Load dataset
    batchSize = 512
    v_batchSize = 512
    device = 'cuda'
    # Load model
    model = CNNModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,4,6,8,10,12],gamma=0.1)
    trainDataset = dataset.MahjongGBDataset(0,1)
    validateDataset = dataset_v.MahjongGBDataset(0,1)
    loader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
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
        if os.path.isfile(logdir + 'checkpoint/model'+ver+'.pkl'):
            print("----------LOAD----------")
            checkpoint = torch.load(logdir + 'checkpoint/model'+ver+'.pkl')
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['acc']
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            epoch = 1  # 如果没进行训练过，初始训练epoch值为1
            best_acc = 0

    acc = 0
    for e in range(20):
        np_loss = np.asarray([])
        np_acc = np.asarray([])
        np_t_acc = np.asarray([])
        np_v_acc = np.asarray([])
        correct_t = 0
        epoch += 1
        print('--------------------epoch--------------------------------')
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(),'mask':d[1].cuda()}}
            logits = model(input_dict)
            tag = d[2].cuda().long()
            loss = F.cross_entropy(logits, tag)
            pred = logits.argmax(dim=1)
            correct_t = torch.eq(pred, d[2].cuda().cuda()).sum().item()
            acc_t = correct_t / len(d[0])
            np_t_acc = np.append(np_t_acc, acc_t)
            if i % 128 == 0 and i != 0 :
                print('Iteration %d/%d' % (i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            np_loss = np.append(np_loss, loss.item())

        sch.step()
        print('Run validation:')
        correct1 = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False,
                          'obs': {'observation': d[0].cuda(),
                                  'mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct1 += torch.eq(pred, d[2].cuda()).sum().item()

        acc = correct1 / len(validateDataset)
        print("acc",acc)

        np_acc = np.append(np_acc, acc)

        if os.path.isfile(logdir + 'checkpoint/acc'+ver+'.npy'):
            all_acc = np.load(logdir + 'checkpoint/acc'+ver+'.npy')
            np_acc = np.append(all_acc, np_acc)
        np.save(logdir + 'checkpoint/acc'+ver+'.npy', np_acc)
        if os.path.isfile(logdir + 'checkpoint/acc_t'+ver+'.npy'):
            all_acc = np.load(logdir + 'checkpoint/acc_t'+ver+'.npy')
            np_t_acc = np.append(all_acc, np_t_acc)
        np.save(logdir + 'checkpoint/acc_t'+ver+'.npy', np_t_acc)

        if os.path.isfile(logdir + 'checkpoint/loss'+ver+'.npy'):
            all_loss = np.load(logdir + 'checkpoint/loss'+ver+'.npy')
            np_loss = np.append(all_loss, np_loss)
        np.save(logdir + 'checkpoint/loss'+ver+'.npy', np_loss)

        checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "acc": acc}
        path_checkpoint = logdir + 'checkpoint/model'+ver+'.pkl'

        if best_acc < acc:
              torch.save(checkpoint, path_checkpoint)

