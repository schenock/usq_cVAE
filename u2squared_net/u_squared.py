import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample_like(src, dst):
    return F.upsample(src, size=dst.shape[2:], mode='bilinear')


class ConvBNReluBlock(nn.Module):
    def __init__(self, input_ch, output_ch, dirate):
        super(ConvBNReluBlock, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1*dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        return self.relu(self.bn(self.conv(image)))


class GenericRSUBlock(nn.Module):
    def __init__(self, input_ch, mid_ch, output_ch, L):
        super(GenericRSUBlock, self).__init__()

        # encoder
        encoder = [ConvBNReluBlock(input_ch=input_ch, output_ch=output_ch, dirate=1)]

        for i in range(L-1):
            dirate = 1  # if i < L-1 else 2
            inp = output_ch if i == 0 else mid_ch
            encoder.append(ConvBNReluBlock(input_ch=inp, output_ch=mid_ch, dirate=dirate))
            if i < L-2:
                encoder.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # TODO: go without defining last layer - use nn.Sequential but carefully and slice
        self.last_layer_enc = ConvBNReluBlock(input_ch=mid_ch, output_ch=mid_ch, dirate=2)

        # decoder
        decoder = []
        for i in range(L - 1):  # decoder has -1 channel
            decoder.append(ConvBNReluBlock(input_ch=mid_ch*2, output_ch=mid_ch, dirate=1))  # TODO: verify mid_ch * 2

        print(nn.Sequential(*encoder, self.last_layer_enc, *decoder))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):



        raise NotImplementedError


if __name__ == '__main__':
    c = GenericRSUBlock(111, 5, 999, 4)