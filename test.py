import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from load_LIDC_data import LIDC_IDRI
from model import ProbabilisticUNet, SegModel, geco_ce
from utils import l2_regularisation

# settings
# segmentation_model = SegModel.U_SQUARED_BIG.value     # Make sure these two
segmentation_model = SegModel.UNET_SIMPLE.value
LOAD_MODEL_FROM = '6_unet_Indep_elbo_nonf_noposw_lat2_b5__epoch0_step532_loss633.pth'  # correspond
dataset_path = 'data/'
num_test_samples = 6
shuffle = True

# data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=dataset_path)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
if shuffle:
    np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]

test_indices = indices[:num_test_samples]
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of test patches:", (len(test_indices)))

# model
net = ProbabilisticUNet(segmentation_model=segmentation_model, input_channels=1, num_classes=1,
                        num_filters=[32, 64, 128, 192], latent_dim=2,
                        num_convs_fcomb=4, beta=2.0)

if LOAD_MODEL_FROM is not None:
    import os
    net.load_state_dict(torch.load(os.path.join(
        "/home/dane/Schen/u2squared-condVAE/saved_checkpoints/",
        LOAD_MODEL_FROM
    )))

net.to(device)
net.eval()

for epoch in range(10):
    for step, (patch, mask, _) in enumerate(test_loader):

        patch, mask = patch.to(device), mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=False)

        masks = net.sample()

        sigma = net.prior_latent_space.stddev.squeeze()
        mu = net.prior_latent_space.mean.squeeze()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(17,17))

        latent_vectors = []
        full = []
        for si in range(-3, 4, 1):
            horizontals = list()
            for sj in range(-3, 4, 1):
                z = list()
                for i in range(2):
                    # i == dim
                    s = si if i == 0 else sj
                    z.insert(i, s * sigma[i] + mu[i])
                # mock 4 dims because of latentdim=6
                # z.extend(mu[2:])
                z = torch.Tensor(z)
                print("z shape", z.shape)
                float_mask = net.sample_for_z([z.unsqueeze(0).to('cuda')])[0].squeeze().squeeze()
                mask_for_z = (torch.sigmoid(float_mask) > 0.5).float()
                numpy_mask = np.array(mask_for_z.detach().cpu())
                if len(horizontals) == 0:
                    horizontals = numpy_mask
                else:
                    horizontals = np.concatenate([horizontals, numpy_mask])
                print()
            if len(full) == 0:
                full = horizontals
            else:
                full = np.hstack([full, horizontals])
        plt.imshow(full)
        plt.scatter(mu[0].detach().cpu(), mu[1].detach().cpu(), c='r', s=3)
        plt.show()


        mask_pred = net.sample(testing=True, mean_sample=True)
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
        mask_pred = torch.squeeze(mask_pred, 0)

        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.imshow(np.array(mask_pred[0].squeeze(0).detach().cpu()))
        plt.title(str(step) + " Reconstruction")

        plt.subplot(132)
        plt.imshow(np.array(mask[0].squeeze(0).detach().cpu()))
        plt.title(str(step) + "GT")

        plt.subplot(133)
        plt.imshow(np.array(patch[0].squeeze(0).detach().cpu()))
        plt.title(str(step) + " Image")
        plt.show()
    exit(1)