import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from load_LIDC_data import LIDC_IDRI, TestLIDC_IDRI
from probabilistic_unet import ProbabilisticUnet

# settings
# segmentation_model = SegModel.U_SQUARED_BIG.value     # Make sure these two
# segmentation_model = SegModel.UNET_SIMPLE.value
LOAD_MODEL_FROM = '1_longrun__epoch0_step2099_loss353.9315185546875.pth'  # correspond
dataset_path = '/home/dane/Schen/u2squared-condVAE/data/'
num_test_samples = 500
shuffle = True

# data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TestLIDC_IDRI(dataset_location=dataset_path)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
if shuffle:
    np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]

eval_indices = indices[:num_test_samples]
# eval_indices = indices
eval_sampler = SubsetRandomSampler(eval_indices)
eval_loader = DataLoader(dataset, batch_size=1, sampler=eval_sampler)
print("Number of test patches:", (len(eval_indices)))

# model
net = ProbabilisticUnet(input_channels=1,
                        num_classes=1,
                        num_filters=[32,64,128,192],
                        latent_dim=2,
                        no_convs_fcomb=4,
                        beta=10.0)

if LOAD_MODEL_FROM is not None:
    import os
    net.load_state_dict(torch.load(os.path.join(
        "./saved_checkpoints/",
        LOAD_MODEL_FROM
    )))

net.to(device)
net.eval()


def energy_distance(seg_samples, gt_seg_modes, num_samples=2):
    num_modes = 4  # fixed for LIDC

    # if num_samples != len(seg_samples) or num_samples != len(gt_seg_modes):
    #     raise ValueError

    d_matrix_YS = np.zeros(shape=(num_modes, num_samples), dtype=np.float32)
    d_matrix_YY = np.zeros(shape=(num_modes, num_modes), dtype=np.float32)
    d_matrix_SS = np.zeros(shape=(num_samples, num_samples), dtype=np.float32)

    # iterate all ground-truth modes
    for mode in range(num_modes):

        ##########################################
        #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
        #   with S ~ P_pred, Y ~ P_gt  			 #
        ##########################################

        # iterate the samples S
        for i in range(num_samples):
            iou = calc_iou(gt_seg_modes[mode], seg_samples[i])
            d_matrix_YS[mode, i] = 1. - iou

        ###########################################
        #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
        #   with Y,Y' ~ P_gt  	   				  #
        ###########################################

        # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
        for mode_2 in range(mode, num_modes):
            iou = calc_iou(gt_seg_modes[mode], gt_seg_modes[mode_2])
            d_matrix_YY[mode, mode_2] = 1. - iou
            d_matrix_YY[mode_2, mode] = 1. - iou

    #########################################
    #   Calculate d(S,S') = 1 - IoU(S,S'),  #
    #   with S,S' ~ P_pred        			#
    #########################################
    # iterate all samples S
    for i in range(num_samples):
        # iterate all samples S'
        for j in range(i, num_samples):
            iou = calc_iou(seg_samples[i], seg_samples[j])
            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    d_SY = (2/(num_modes * num_samples)) * np.sum(d_matrix_YS)
    d_SS = (1/num_samples**2) * np.sum(d_matrix_SS)
    d_YY = (1/num_modes**2) * np.sum(d_matrix_YY)

    e_dist = d_SY - d_SS - d_YY
    # print("dist: ", e_dist)
    return e_dist


def calc_iou(x, y):
    return iou_pytorch(x, y)


SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    # print("iou: ", thresholded)
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


NUM_SAMPLES = 2
NUM_MODES = 4


def _calc_ged(s):
    list_ged = []
    for step, (patch, mask, _) in enumerate(eval_loader):

        patch, mask = patch.to(device), mask.to(device)
        net.forward(patch, None, training=False)

        # sample `num_samples` times
        samples = np.array([
            (torch.sigmoid(net.sample(testing=True)) > 0.5).int().squeeze(0) for _ in range(s)])

        energy = energy_distance(seg_samples=samples, gt_seg_modes=mask.squeeze(0).unsqueeze(1).int(), num_samples=s)
        # this is one point in the energy plot for sample idx:step
        list_ged.append(energy)

    return list_ged


def seaborn_plot_ged():
    e_distances = []
    e_means = []
    samples_column = []

    for s in [1, 4, 8, 16]:
        ged = _calc_ged(s=s)
        # ged = ged[~np.isnan(ged)]

        e_distances.extend(ged)
        samples_column.extend([s] * len(ged))


    import pandas as pd
    energy = pd.DataFrame(data={'energy': e_distances, 'num_samples': samples_column})

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    f = plt.figure(figsize=(5, 5), dpi=150)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    sns.stripplot(x="num_samples", y="energy", data=energy, alpha=0.5, s=2, ax=ax)
    # sns.stripplot(x="num_samples", y="energy", data=means, s=18, marker='^', color='k', ax=ax, jitter=False)
    # sns.stripplot(x="num_samples", y="energy", data=means, s=14, marker='^', ax=ax, jitter=False)
    ax.set_title('Probabilistic U-Net (validation set)', y=1.03)
    fs = 12
    ax.set_ylabel(r'$D_{ged}^{2}$', fontsize=fs)
    ax.set_xlabel('# samples', fontsize=fs)
    plt.show()



seaborn_plot_ged()
