import os

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from nath_data_loader import Rescale
from nath_data_loader import RescaleT
from nath_data_loader import RandomCrop
from nath_data_loader import ToTensor
from nath_data_loader import ToTensorLab
from nath_data_loader import SalObjDataset
from model import U2SquaredNet

# Configs
from u2_dane.model import BigU2Net

model_name = 'big_u2net_dane'

data_dir = "/home/dane/Schen/Data/DUTS-TR/"
tra_image_dir = "DUTS-TR-Image/"
tra_label_dir = "DUTS-TR-Mask/"

image_ext = '.jpg'
label_ext = '.png'

# Training configs
epoch_num = 2000
batch_size_train = 4
batch_size_val = 1
train_num = 0
val_num = 0
learning_rate = 0.001

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 1800  # save the model every 2000 iterations

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)


def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data, loss6.data))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
# define the net
if model_name == 'u2net_dane':
    net = U2SquaredNet(3, 1)
elif model_name == 'big_u2net_dane':
    net = BigU2Net(3, 1)
else:
    raise NotImplementedError()

if torch.cuda.is_available():
    print("CUDA available.")
    net.cuda()
else:
    print("CUDA NOT AVAILABLE.")

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(iter(salobj_dataloader)):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            save_folder = os.path.join(model_dir, model_name)
            if not os.path.exists(save_folder):
                print("Folder does not exist. Creating folder: {}".format(save_folder))
                os.makedirs(save_folder)

            torch.save(net.state_dict(), save_folder + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

if __name__ == "__main__":
    print()
