import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter

from load_LIDC_data import LIDC_IDRI
from model import ProbabilisticUNet, SegModel, geco_ce
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
train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

import visdom
vis = visdom.Visdom(port=7756)
# assert vis.check_connection()
loss_window = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='Loss', title='unet training loss', legend=['Loss']))

loss_window_reconst = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='Loss', title='unet constraint: reconstruct loss',
              legend=['Constraint: Reconstruction Loss']))

loss_window_kl = vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='b * KL', title='unet kl value', legend=['beta * KL Value']))

loss_window_elbo= vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='-elbo', title='unet elbo', legend=['ELBO']))

loss_window_reg= vis.line(
    Y=torch.zeros((1)).cpu(),
    X=torch.zeros((1)).cpu(),
    opts=dict(xlabel='epoch', ylabel='reg_loss', title='dane l2 loss', legend=['L2 Loss']))


# TODO: Move to configs
segmentation_model = SegModel.U_SQUARED_SMALL.value
# segmentation_model = SegModel.UNET_SIMPLE.value

net = ProbabilisticUNet(segmentation_model=segmentation_model, input_channels=1, num_classes=1,
                        num_filters=[32, 64, 128, 192], latent_dim=6,
                        num_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
loss_fn = 'geco_ce'  # geco, cross-entropy
epochs = 10
lambd_init = torch.FloatTensor([1]).to(device)

_init_geco_params = {
    'alpha': 0.99,
    'lambd_step': 100,
}

alpha = 0.99
lambda_step = 100
lambda_init = torch.FloatTensor([1])

lambd = lambd_init

for epoch in range(epochs):
    for step, (patch, mask, _) in enumerate(train_loader):
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)

        if loss_fn == 'elbo_ce':
            elbo, reconstruction_loss, kl_scaled, beta = net.elbo(mask, step=step)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            print("reg: ", reg_loss)
            loss = -elbo + 2e-4 * reg_loss
            # tb.add_scalar("Loss", loss, step)
            print("Loss: ", loss, ", reconst_loss:", reconstruction_loss, ", kl:", kl_scaled)
        elif loss_fn == 'geco_ce':
            KL, constraint = net.compute_KL_CE(mask, step=step)
            geco_ce_loss = geco_ce(KL, constraint, lambd)
            loss = geco_ce_loss.to(device)
            print("Loss: ", loss, ", lmbd: ", lambd, ", CE_loss:", constraint, ", KL:", KL)
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        # noinspection PyUnboundLocalVariable
        loss.backward()
        optimizer.step()

        if 'geco' in loss_fn:
            with torch.no_grad():
                if epoch == 0 and step == 0:
                    # noinspection PyUnboundLocalVariable
                    constrain_ma = constraint
                else:
                    constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constraint
                    if step % lambda_step == 0:
                        #                     print(torch.exp(constrain_ma), lambd)
                        lambd *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.1)

        # visualize
        vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                 win=loss_window,
                 update='append')

        # elbo loss
        # vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([-elbo]).unsqueeze(0).cpu(),
        #          win=loss_window_elbo,
        #          update='append')

        # reconstruction loss
        vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([constraint]).unsqueeze(0).cpu(),
                 win=loss_window_reconst,
                 update='append')

        # KL
        vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([KL]).unsqueeze(0).cpu(),
                 win=loss_window_kl,
                 update='append')

        # l2
        # vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([reg_loss]).unsqueeze(0).cpu(),
        #          win=loss_window_reg,
        #          update='append')
