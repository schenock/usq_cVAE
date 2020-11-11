from enum import Enum, unique

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, MultivariateNormal, kl

from utils import init_weights, init_weights_orthogonal_normal
from unet_simple.unet import Unet

sys_wide_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    r"""
    A CNN encoder built from `len(num_filters)` x a block of `num_convs_per_block` convolutional layers, after each
    a pooling operation is called. A RELU activation is used after each conv layer.
    """

    def __init__(self,
                 input_ch,
                 num_filters,
                 num_convs_per_block,
                 initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.input_ch = input_ch
        self.num_filters = num_filters
        self.num_convs_per_block = num_convs_per_block

        self.layers = self._construct_net(num_filters, num_convs_per_block, padding, posterior)
        self.layers.apply(init_weights)

    def _construct_net(self, num_filters, num_convs_per_block, padding, posterior):
        # TODO: 1 Check this
        if posterior:
            self.input_ch += 1

        layers = []
        # TODO: Refactor this (in pythonic way)
        for i in range(len(num_filters)):
            """
            for each block
            """
            input_dim = self.input_ch if i == 0 else output_dim
            output_dim = num_filters[i]
            print("input dim: {}, output dim {}".format(input_dim, output_dim))

            # after each one pool
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, inp):
        return self.layers(inp)


class AxisAlignedConvGaussian(nn.Module):
    r"""
     Convolutional network that parametrizes a Gaussian with axis aligned covariance matrix.
     """

    def __init__(self,
                 input_ch,
                 num_filters,
                 num_convs_per_block,
                 latent_dim,
                 initializers,
                 posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.channel_axis = 1  # TODO: what is this?
        self.input_ch = input_ch
        self.num_filters = num_filters
        self.num_convs_per_block = num_convs_per_block

        self.latent_dim = latent_dim
        self.posterior = posterior
        self.name = "Posterior" if self.posterior else "Prior"

        self.encoder = \
            Encoder(self.input_ch, self.num_filters, self.num_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * latent_dim, kernel_size=(1, 1), stride=1)

        # --- what is this ? ---
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        # TODO: Move these to the init method in utils
        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segmentation=None):
        # print("forwards")
        if segmentation is not None:
            self.show_img = input
            self.show_seg = segmentation
            input = torch.cat((input, segmentation), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # compute mean of encoding
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # push encoding through 1 layer nn to convert it to 2 x latent_dim shape
        mu_log_sigma = self.conv_layer(encoding)

        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # TODO: Change to MultivatiateNormal (assuming it is identical to Independent(Normal, 1)
        # diag_cov_multiv_dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

        cov = torch.stack([torch.diag(sigma) for sigma in torch.exp(log_sigma)])
        diag_cov_mvn = MultivariateNormal(mu, cov)

        return diag_cov_mvn


class FComb(nn.Module):
    r"""
    Combines sample taken from latent space to output of U^2 net.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, num_convs_fcomb, initializers,
                 use_tile=True):
        super(FComb, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.num_convs_fcomb = num_convs_fcomb
        self.use_tile = use_tile
        self.name = 'FComb'

        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        if self.use_tile:
            layers = \
                [nn.Conv2d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1),
                 nn.ReLU(inplace=True)]

            for _ in range(num_convs_fcomb - 2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1))

            self.layers = nn.Sequential(*layers)

            init_func = init_weights_orthogonal_normal if initializers['w'] == 'orthogonal' else init_weights
            self.layers.apply(init_func)

    def forward(self, feature_map, z):
        r"""
        feature_map: the output feature map from U^2 net
        z: (batch_size x latent_dim) latent vector

        So broadcast Z to batch_sizexlatent_dimxHxW.
        """
        if self.use_tile:
            # TODO: (z, 1) should be batch size dimension?
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, dim=2, n_tile=feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, dim=3, n_tile=feature_map.shape[self.spatial_axes[1]])

            concat = torch.cat((feature_map, z), dim=self.channel_axis)
            return self.layers(concat)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            sys_wide_device)
        return torch.index_select(a, dim, order_index)


@unique
class SegModel(Enum):
    UNET_SIMPLE = 0
    U_SQUARED_SMALL = 1
    U_SQUARED_BIG = 2
    U_SQUARED_SIMPLE = None


class ProbabilisticUNet(nn.Module):
    r"""

    """

    def __init__(self, segmentation_model=0, input_channels=1, num_classes=1,
                 num_filters=(32, 64, 128, 192), latent_dim=6, num_convs_fcomb=4,
                 beta=10.0):
        super(ProbabilisticUNet, self).__init__()
        self.segmentation_model = segmentation_model
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.num_convs_fcomb = num_convs_fcomb
        self.beta = beta

        self.num_convs_per_block = 3
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        # TODO: Clarify why he normal init here ? Read about difference.
        self.beta = beta
        self.z_prior_sample = 0

        if self.segmentation_model == SegModel.UNET_SIMPLE.value:

            self.unet = Unet(self.input_channels,
                             self.num_classes,
                             self.num_filters,
                             self.initializers,
                             apply_last_layer=False, padding=True).to(sys_wide_device)
        elif self.segmentation_model == SegModel.U_SQUARED_SMALL.value:
            from u2_dane.model import U2SquaredNet
            from u2_dane.model import U2SquaredNetSmall
            self.unet = U2SquaredNetSmall(in_ch=1, out_ch=1, mid_ch=32).to(sys_wide_device)  # input channels = 1 grayscale inputs
        elif self.segmentation_model == SegModel.U_SQUARED_BIG.value:
            from u2_dane.model import BigU2Net
            self.unet = BigU2Net(1, 1)
        else:
            raise NotImplementedError

        encoder_args = \
            (self.input_channels, self.num_filters, self.num_convs_per_block, self.latent_dim, self.initializers)

        self.prior = AxisAlignedConvGaussian(*encoder_args, posterior=False).to(sys_wide_device)
        self.posterior = AxisAlignedConvGaussian(*encoder_args, posterior=True).to(sys_wide_device)
        self.fcomb = FComb(self.num_filters,
                           self.latent_dim,
                           self.input_channels,
                           self.num_classes,
                           self.num_convs_fcomb,
                           {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(sys_wide_device)
        # TODO: Clarify why orthogonal init here?

    def forward(self, patch, mask, training=True):
        r"""

        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, mask)
        self.prior_latent_space = self.prior.forward(patch)

        if self.segmentation_model == SegModel.U_SQUARED_SMALL.value or \
                self.segmentation_model == SegModel.U_SQUARED_BIG.value:
            self.unet_features = self.unet.forward(patch)
        elif self.segmentation_model == SegModel.UNET_SIMPLE.value:
            self.unet_features = self.unet.forward(patch, False)
        print()

    def sample(self, testing=False):
        r"""
        Sample a segmentation by reconstructing from a prior sample from latent dist
        and combining this with UNet features        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean, calculate_posterior=False, z_posterior=None):
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        # TODO: Check what is the difference between loc and sample and rsample.
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=False):
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, mask, analytic_kl=True, reconstruct_posterior_mean=False, step=None):
        r"""
        Evidence lower bound of the log-likelihood or P(Y|X)
        """

        z_posterior = self.posterior_latent_space.rsample()
        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = \
            self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False,
                             z_posterior=z_posterior)

        criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)

        # ---------
        if step > 70 and step % 20 == 0:
            import matplotlib.pyplot as plt
            plt.imshow(np.array(self.reconstruction[0].squeeze(0).detach().cpu()))
            plt.title("reconstruction")
            plt.show()
            plt.imshow(np.array(mask[0].squeeze(0).detach().cpu()))
            plt.title("target")
            plt.show()

        reconstruction_loss = criterion(input=self.reconstruction, target=mask)

        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl), self.reconstruction_loss, self.beta * self.kl, self.beta

    def compute_KL_CE(self, mask, analytic_kl=True, reconstruct_posterior_mean=False, step=None):
        r"""
        computes KL divergence and cross entropy loss.
        """
        z_posterior = self.posterior_latent_space.rsample()
        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = \
            self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False,
                             z_posterior=z_posterior)

        criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)

        reconstruction_loss = criterion(input=self.reconstruction, target=mask)

        self.sum_reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        if step > 70 and step % 20 == 0:
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.imshow(np.array(self.reconstruction[0].squeeze(0).detach().cpu()))
            plt.title("reconstruction")

            plt.subplot(212)
            plt.imshow(np.array(mask[0].squeeze(0).detach().cpu()))
            plt.title("target")

            plt.show()

        return self.kl, self.mean_reconstruction_loss


def geco_ce(KL, mean_reconstruction_loss, lambd):
    return KL + lambd * mean_reconstruction_loss
