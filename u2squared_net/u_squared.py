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

        self._L = L

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

        # print(nn.Sequential(*encoder, self.last_layer_enc, *decoder))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        outputs = []
        downward = x

        # TODO: Replace downward with downstream
        # encoder
        for layer in self.encoder:
            downward = layer(downward)
            if isinstance(layer, ConvBNReluBlock):
                outputs.append(downward)

        hx_in = outputs.pop(0).clone()  # TODO: clone might be unnecessary here
        assert len(self.decoder) == len(outputs)
        # decoder
        upward = self.last_layer_enc.forward(downward)
        for layer in self.decoder[:-1]:
            upward = layer(torch.cat((outputs.pop(), upward), 1))
            upward = _upsample_like(upward, outputs[-1])

        return hx_in + self.decoder[-1](torch.cat((outputs.pop(), upward), 1))


class GenericRSUFBlock(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, L=4):

        super(GenericRSUFBlock, self).__init__()

        encoder = [ConvBNReluBlock(input_ch=in_ch, output_ch=out_ch, dirate=1)]
        decoder = []

        for i in range(L):
            # print("enc: ", 2 ** i)
            encoder.append(ConvBNReluBlock(input_ch=out_ch if i == 0 else mid_ch, output_ch=mid_ch, dirate=2 ** i))

        for i in range(L-1):
            # print(i)
            # print("dec: ", 2 ** (L-2-i))
            decoder.append(ConvBNReluBlock(input_ch=mid_ch * 2, output_ch=mid_ch, dirate=2 ** (L-2-i)))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        outputs = []
        downward = x
        for layer in self.encoder:
            downward = layer(downward)
            outputs.append(downward)

        hx_in = outputs.pop(0).clone()
        upward = outputs.pop()
        for layer in self.decoder:
            upward = layer(torch.cat((outputs.pop(), upward), 1))

        return hx_in + upward


class U2SquaredNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2SquaredNet, self).__init__()

        # encoder
        self.stage1 = GenericRSUBlock(in_ch, 16, 64, L=7)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = GenericRSUBlock(64, 16, 64, L=6)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = GenericRSUBlock(64, 16, 64, L=5)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = GenericRSUBlock(64, 16, 64, L=4)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = GenericRSUFBlock(64, 16, 64, L=4)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = GenericRSUFBlock(64, 16, 64, L=4)

        # decoder
        self.stage5d = GenericRSUFBlock(128, 16, 64, L=4)
        self.stage4d = GenericRSUBlock(128, 16, 64, L=4)
        self.stage3d = GenericRSUBlock(128, 16, 64, L=5)
        self.stage4d = GenericRSUBlock(128, 16, 64, L=6)
        self.stage1d = GenericRSUBlock(128, 16, 64, L=7)

        self.out_conv = nn.Conv2d(6, out_ch, 1)


if __name__ == '__main__':
    print()
    # Test blocks
    # first RSU block, and going down to a u-net of GenRSUblocks.
    # c1 = GenericRSUBlock(3, 32, 64, L=4)
    # c1f = GenericRSUFBlock(3, 32, 64, L=4)
    # c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))
    # c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))

    # U-squared Generic

