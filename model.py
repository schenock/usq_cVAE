import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from utils import init_weights

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

            # after each one pool
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs_per_block-1):
                layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
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

        self.channel_axis = 1  #TODO: what is this?
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

            # TODO: Check all this logic, and inputs outputs
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

            mu = mu_log_sigma[:, :self.latent_dim]
            log_sigma = mu_log_sigma[:, self.latent_dim:]

            # TODO: Explore this!
            diag_cov_multiv_dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
            return diag_cov_multiv_dist





























