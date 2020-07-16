import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample_like(src, dst):
    return F.upsample(src, size=dst.shape[2:], mode='bilinear')


class ConvBNReluBlock(nn.Module):
    r"""
    Conv, BatchNorm, ReLU operations encapsulated in a block.
    """
    def __init__(self, input_ch, output_ch, dirate):
        super(ConvBNReluBlock, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1*dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        return self.relu(self.bn(self.conv(image)))


class GenericRSUBlock(nn.Module):
    r"""
    Generic RSU Block, parametrized by `L`, the parameter defines depth.
    RSU block as proposed in <TODO: add link>, which has a U-net like architecture

    """
    def __init__(self, input_ch=3, mid_ch=12, output_ch=3, L=None):
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
        for i in range(L - 2):  # decoder has -1 channel, -1 for the last channel out
            decoder.append(ConvBNReluBlock(input_ch=mid_ch*2, output_ch=mid_ch, dirate=1))  # TODO: verify mid_ch * 2

        decoder.append(ConvBNReluBlock(input_ch=mid_ch*2, output_ch=output_ch, dirate=1))  # TODO: verify mid_ch * 2

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def __repr__(self):
        return "Block RSU: " + str(self._L)

    def forward(self, x):
        outputs = []
        downward = x

        # TODO: Replace downward with downstream
        # encoder
        for layer in self.encoder:
            downward = layer(downward)
            if isinstance(layer, ConvBNReluBlock):
                outputs.append(downward)

        hx_in = outputs.pop(0).clone()
        assert len(self.decoder) == len(outputs)
        # decoder
        upward = self.last_layer_enc.forward(downward)
        for layer in self.decoder[:-1]:
            upward = layer(torch.cat((upward, outputs.pop()), 1))
            upward = _upsample_like(upward, outputs[-1])

        return hx_in + self.decoder[-1](torch.cat((upward, outputs.pop()), 1))


class GenericRSUFBlock(nn.Module):
    r"""
    Generic RSUF block. A modification of the original RSU block, with dilated convolutions.
    The parameter `dirate` controls the sparsity/density of the dilatations.

    """
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, L=4):

        super(GenericRSUFBlock, self).__init__()
        encoder = [ConvBNReluBlock(input_ch=in_ch, output_ch=out_ch, dirate=1)]
        decoder = []

        for i in range(L):
            encoder.append(
                ConvBNReluBlock(input_ch=out_ch if i == 0 else mid_ch, output_ch=mid_ch, dirate=2 ** i)
            )

        for i in range(L-2):
            decoder.append(
                ConvBNReluBlock(input_ch=mid_ch * 2, output_ch=mid_ch, dirate=2 ** (L-2-i))
            )

        i += 1
        decoder.append(ConvBNReluBlock(input_ch=mid_ch * 2, output_ch=out_ch, dirate=2 ** (L-2-i)))
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
        self.stage2d = GenericRSUBlock(128, 16, 64, L=6)
        self.stage1d = GenericRSUBlock(128, 16, 64, L=7)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.out_conv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        hx1 = self.stage1.forward(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2.forward(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3.forward(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4.forward(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5.forward(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6.forward(hx)
        hx6_up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d.forward(torch.cat((hx6_up, hx5), 1))
        hx5d_up = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d.forward(torch.cat((hx5d_up, hx4), 1))
        hx4d_up = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d.forward(torch.cat((hx4d_up, hx3), 1))
        hx3d_up = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d.forward(torch.cat((hx3d_up, hx2), 1))
        hx2d_up = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d.forward(torch.cat((hx2d_up, hx1), 1))

        # side output TODO: What are side outputs used for?
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.out_conv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


class BigU2Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(BigU2Net, self).__init__()

        self.stage1 = GenericRSUBlock(in_ch, 32, 64, L=7)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = GenericRSUBlock(64, 32, 128, L=6)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = GenericRSUBlock(128, 64, 256, L=5)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = GenericRSUBlock(256, 128, 512, L=4)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = GenericRSUFBlock(512, 256, 512, L=4)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = GenericRSUFBlock(512, 256, 512, L=4)

        # dec
        self.stage5d = GenericRSUFBlock(1024, 256, 512, L=4)
        self.stage4d = GenericRSUBlock(1024, 128, 256, L=4)
        self.stage3d = GenericRSUFBlock(512, 64, 128, L=5)
        self.stage2d = GenericRSUFBlock(256, 32, 64, L=6)
        self.stage1d = GenericRSUFBlock(128, 16, 64, L=7)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


if __name__ == '__main__':
    print()
    # Test blocks
    # first RSU block, and going down to a u-net of GenRSUblocks.
    # c1 = GenericRSUBlock(3, 32, 64, L=4)
    # c1f = GenericRSUFBlock(3, 32, 64, L=4)
    # c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))
    # c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))

    # U-squared Generic


