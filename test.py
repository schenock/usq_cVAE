import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from load_LIDC_data import LIDC_IDRI
from model import ProbabilisticUNet, SegModel, geco_ce
from utils import l2_regularisation

# settings
segmentation_model = SegModel.U_SQUARED_BIG.value     # Make sure these two
LOAD_MODEL_FROM = '1_punet_bigu_Indep_focal_posw15_'  # correspond
dataset_path = 'data/'
num_test_samples = 10
shuffle = False

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
                        num_filters=[32, 64, 128, 192], latent_dim=6,
                        num_convs_fcomb=4, beta=10.0)

if LOAD_MODEL_FROM is not None:
    net.load_state_dict(torch.load(
        "/home/dane/Schen/u2squared-condVAE/saved_checkpoints/1_punet_bigu_Indep_focal_posw15__epoch0_step2606_loss01023.pth"))

net.to(device)
net.eval()

for epoch in range(10):
    for step, (patch, mask, _) in enumerate(test_loader):

        patch, mask = patch.to(device), mask.to(device)
        mask = torch.unsqueeze(mask, 1)
        net.forward(patch, mask, training=True)

        masks = net.sample()

        sigma = net.prior_latent_space.stddev.squeeze()
        mu = net.prior_latent_space.mean.squeeze()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,12))

        latent_vectors = []
        full = []
        for si in range(-20, 20, 1):
            horizontals = list()
            for sj in range(-20, 20, 1):
                z = list()
                for i in range(2):
                    # i == dim
                    s = si if i == 0 else sj
                    z.insert(i, s * sigma[i] + mu[i])
                # mock 4 dims because of latentdim=6
                z.extend(mu[2:])
                z = torch.Tensor(z)
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
        plt.show()


        mask_pred = net.sample(testing=True, mean_sample=True)
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
        mask_pred = torch.squeeze(mask_pred, 0)

        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(np.array(mask_pred[0].squeeze(0).detach().cpu()))
        plt.title(str(step) + " Reconstruction")

        plt.subplot(122)
        plt.imshow(np.array(mask[0].squeeze(0).detach().cpu()))
        plt.title(str(step) + "GT")
        plt.show()

    exit(1)