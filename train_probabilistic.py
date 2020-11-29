import os
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter
from experiment_utils import setup_experiment
from load_LIDC_data import LIDC_IDRI
from model import ProbabilisticUNet, SegModel, geco_ce
from utils import l2_regularisation
from easydict import EasyDict as edict


settings = edict()
settings.visualize = False
settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
settings.dataset_location = 'data/'
settings.segmentation_model = SegModel.UNET_SIMPLE.value
# segmentation_model = SegModel.U_SQUARED_SMALL.value
# segmentation_model = SegModel.U_SQUARED_BIG.value

settings.SAVE_AT_STEP = 250
settings.EXP_NAME = '1_test_model'
settings.ckpts_folder = './saved_checkpoints'

# TODO Add all geco or loss_fn params to settings + distrib params.
settings.latent_distribution = None
settings.loss = None
settings.pos_weighting = None
settings.loss_params = None

settings.use_lr_scheduler = False
settings.batch_size = 5
settings.latent_dim = 2
settings.beta = 2.0

settings.visdom_port = 7789

# Data
dataset = LIDC_IDRI(dataset_location=settings.dataset_location)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=settings.batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices), len(test_indices)))

if settings.visualize:
    import visdom
    vis = visdom.Visdom(port=settings.visdom_port)
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


device = torch.device(settings.device)

net = ProbabilisticUNet(segmentation_model=settings.segmentation_model, input_channels=1, num_classes=1,
                        num_filters=[32, 64, 128, 192], latent_dim=settings.latent_dim,
                        num_convs_fcomb=4, beta=settings.beta)
net.to(settings.device)


setup_experiment(exp_name=settings.EXP_NAME,
                 exp_params=dict(settings),
                 net_params=net.model_description(),
                 ckpts_folder=settings.ckpts_folder)


optimizer = torch.optim.Adam(net.parameters(), lr=0.8e-4, weight_decay=0)
if settings.use_lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

loss_fn = 'elbo_ce'  # geco, cross-entropy
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
        # # eval
        # net.load_state_dict(torch.load("/home/dane/Schen/u2squared-condVAE/saved_checkpoints/1_punet_bigu_Indep_focal_posw15__epoch0_step2606_loss01023.pth"))

        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)

        if step > 100 and step % 80 == 0:
            num_preds = 2
            predictions = []
            # for i in range(num_preds):
            mask_pred = net.sample(testing=True, mean_sample=True)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            mask_pred = torch.squeeze(mask_pred, 0)
            # predictions.append(mask_pred)
            for i in range(num_preds):  # batch size
                import matplotlib.pyplot as plt
                plt.subplot(121)
                plt.imshow(np.array(mask_pred[i].squeeze(0).detach().cpu()))
                plt.title("Reconstruction")

                plt.subplot(122)
                plt.imshow(np.array(mask[i].squeeze(0).detach().cpu()))
                plt.title("GT")
                plt.show()

            # predictions = torch.cat(predictions, 0)

        # save model
        if ((epoch + 1) * len(train_loader) + step) % settings.SAVE_AT_STEP == 0 and step != 0:
            print("Saving model at: {}. Loss: {}".format(
                ((epoch + 1) * len(train_loader) + step),
                loss
            ))
            model_save_path = os.path.join(settings.ckpts_folder,
                                           settings.EXP_NAME,
                                           "model_epoch_{}_step_{}_loss_{}.pth".format(epoch, step, loss))
            torch.save(net.state_dict(), model_save_path)

        if loss_fn == 'elbo_ce':
            elbo, reconstruction_loss, kl_scaled, beta = net.elbo(mask, step=step)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            print("reg: ", reg_loss)
            loss = -elbo + 1e-5 * reg_loss
            # tb.add_scalar("Loss", loss, step)
            print("Loss: ", loss, ", reconst_loss:", reconstruction_loss, ", kl:", kl_scaled)
        elif loss_fn == 'geco_ce':
            KL, constraint = net.compute_KL_CE(mask, focal=True, step=step)
            geco_ce_loss = geco_ce(KL, constraint, lambd)
            loss = geco_ce_loss.to(device)
            print("Loss: ", loss, ", lmbd: ", lambd, ", CE_loss:", constraint, ", KL:", KL)
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        # noinspection PyUnboundLocalVariable
        loss.backward()
        optimizer.step()

        if settings.use_lr_scheduler:
            scheduler.step(loss)

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

        if settings.visualize:
            # visualize
            vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                     win=loss_window,
                     update='append')

            # elbo loss
            vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([-elbo]).unsqueeze(0).cpu(),
                     win=loss_window_elbo,
                     update='append')

            # reconstruction loss
            # vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([constraint]).unsqueeze(0).cpu(),
            #          win=loss_window_reconst,
            #          update='append')
            #
            # # KL
            KL = kl_scaled * beta
            vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([KL]).unsqueeze(0).cpu(),
                     win=loss_window_kl,
                     update='append')

            # l2
            # vis.line(X=torch.ones((1, 1)).cpu() * step, Y=torch.Tensor([reg_loss]).unsqueeze(0).cpu(),
            #          win=loss_window_reg,
            #          update='append')
