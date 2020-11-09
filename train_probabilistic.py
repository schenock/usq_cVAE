import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter

from load_LIDC_data import LIDC_IDRI
from model import ProbabilisticUNet, SegModel
from utils import l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location='data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

# TODO: Move to configs
segmentation_model = SegModel.U_SQUARED_SMALL.value

net = ProbabilisticUNet(segmentation_model=segmentation_model, input_channels=1, num_classes=1,
                        num_filters=[32, 64, 128, 192], latent_dim=2,
                        num_convs_fcomb=4, beta=1.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)
epochs = 10

# tb = SummaryWriter()

import visdom

vis = visdom.Visdom(port=7755)
assert vis.check_connection()
loss_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='Loss', title='dane training loss', legend=['Loss']))

loss_window_reconst = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='Loss', title='dane reconstruct loss', legend=['Reconstruction Loss']))

for epoch in range(epochs):
    for step, (patch, mask, _) in enumerate(train_loader):
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)
        elbo, reconstruction_loss, kl, beta = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        # tb.add_scalar("Loss", loss, step)
        print("Loss: ", loss, ", reconst_loss:", reconstruction_loss, ", kl:", kl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([loss]).unsqueeze(0).cpu(), win=loss_window,
                 update='append')

        # reconstruction loss
        vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([reconstruction_loss]).unsqueeze(0).cpu(), win=loss_window,
                 update='append')
