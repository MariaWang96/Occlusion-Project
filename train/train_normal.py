import torch
import torch.nn as nn
import os
import torchvision.transforms as transfomrs
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from model.mobilenet_v1 import mobilenet_1
from dataloader import TrainValDataset_Normal, ToTensorG, NormalizeG, save_checkpoint

from losses.step3_wpdc_vdc_nwlmdc import Loss_all

train_loss_plot = []
val_loss_plot = []
worst_loss = np.inf


def train(train_dataloader, model, criterion, optimizer, epoch):
    global train_loss_plot
    losses = []
    model.train()

    with tqdm(total=len(train_dataloader)) as tqm:
        for i, (input, target, normals_weight) in enumerate(train_dataloader):
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            normals_weight.requires_grad = False
            normals_weight = normals_weight.cuda(non_blocking=True)
            output = model(input.cuda())

            loss = criterion(output, target.cuda(), normals_weight.cuda())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqm.set_description('Iter %i' % (i))
            tqm.set_postfix(loss=loss.item())
            tqm.update()
    tqm.close()
    print('Epoch:{}, Loss_avg: {:.4f}'.format(epoch, np.mean(losses)))

    train_loss_plot.append(np.mean(losses))


def validate(val_dataloader, model, criterion, scheduler, epoch): # scheduler
    global val_loss_plot
    global worst_loss
    model.eval()
    scheduler.step()

    with torch.no_grad():
        losses = []
        for i, (input, target, normals_weight) in enumerate(val_dataloader):
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            normals_weight.requires_grad = False
            normals_weight = normals_weight.cuda(non_blocking=True)
            output = model(input.cuda())

            loss = criterion(output, target.cuda(), normals_weight.cuda())
            losses.append(loss.item())

        val_loss = np.mean(losses)
        print('Epoch:{}, Val Loss:{:.4f}'.format(epoch, val_loss))
        val_loss_plot.append(val_loss)

        if val_loss < worst_loss:
            filename = '../weights/phase3_wpdc_vcd_nwlmdc_3_1.pth.tar'
            save_checkpoint(
                {'state_dict': model.state_dict()},
                filename
            )
            worst_loss = min(val_loss_plot)
            print('worst_loss: {:.4f}'.format(worst_loss))

def main():

    base_lr = 1e-4
    batch_size = 64
    num_workers = 24
    start_epoch = 1
    epoches = 50
    pretrained = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = mobilenet_1()

    # if pretrained:
    #     checkpoint_fp = '../weights/mb1_step1_1_wpdc_50.pth.tar'
    #     checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    #     model.load_state_dict(checkpoint)
    if pretrained:
        checkpoint_fp = '../weights/phase3_wpdc_vcd_nwlmdc_1_1.pth.tar'
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)

    model = model.to(device).cuda()

    criterion = Loss_all().cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)  # filter(lambda p: p.requires_grad, model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # data
    normalize = NormalizeG(mean=127.5, std=128)

    root = '3D-Dataset/300W-LP-Aug-Occlusion'

    filelists_train = '../train_configs_aug/train_aug_120x120_train_aug.txt'
    param_fp_train = '../train_configs_aug/param_all_norm_train_aug.pkl'
    normal_train = '../train_configs_aug/normals_weight_68_train_aug.pkl'

    filelists_val = '../train_configs_aug/train_aug_120x120_val_aug.txt'
    param_fp_val = '../train_configs_aug/param_all_norm_val_aug.pkl'
    normal_val = '../train_configs_aug/normals_weight_68_val_aug.pkl'

    train_dataset = TrainValDataset_Normal(root=root,
                                    filelists=filelists_train,
                                    param_fp=param_fp_train,
                                    wnormal_pth= normal_train,
                                    transform=transfomrs.Compose([ToTensorG(), normalize]))
    val_dataset = TrainValDataset_Normal(root=root,
                                    filelists=filelists_val,
                                    param_fp=param_fp_val,
                                    wnormal_pth= normal_val,
                                    transform=transfomrs.Compose([ToTensorG(), normalize]))
    train_loader = DataLoader(train_dataset,
                              batch_size= batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                              batch_size= batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True)

    cudnn.benchmark = True

    for epoch in range(start_epoch, epoches+1):
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion, scheduler, epoch) # scheduler

    # visualize the training process
    fig = plt.figure(figsize=(4, 1), dpi=160)
    ax1 = fig.add_subplot(1, 1, 1)
    x = range(0, epoches)
    # plot
    ax1.plot(x, train_loss_plot, color='green', label="Train loss")
    ax1.plot(x, val_loss_plot, color='blue', label="Validate loss")
    ax1.set_xticks(x[::1])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Val Loss")
    ax1.grid(alpha=0.1, linestyle="--")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()