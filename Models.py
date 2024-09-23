import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# refer to https://github.com/virylon/AMU-Net
class BasicConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.bc = nn.Sequential(
            nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.bc(x)
        return x


class Attention(nn.Module):
    """input_channels, out_channels"""

    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()

        sub_channels = int(out_channels / 4)
        self.q = BasicConv2d(in_channels, sub_channels, 1)
        self.k = BasicConv2d(in_channels, sub_channels, 1)
        self.v = BasicConv2d(in_channels, out_channels, 3, padding=1)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            BasicConv2d(sub_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        psi = self.psi(q*k)
        out = psi*v

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Attention(in_channels, out_channels),
            Attention(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_noAttention(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_noAttention, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Multiscale(nn.Module):
    def __init__(self, in_channels):
        super(Multiscale, self).__init__()
        sub_channels = int(in_channels/4)
        self.s0 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        s0 = self.s0(x)
        s1 = self.s1(x)
        s2 = self.s2(x)
        s3 = self.s3(x)
        cats = torch.cat((s0,s1,s2,s3),dim=1)

        return cats


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, attention=True):
        super().__init__()
        if attention:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv_noAttention(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,
                 bilinear=False, multiscale=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ms = Multiscale(out_channels) if multiscale else lambda x: x
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        ms = self.ms(x2)
        x = torch.cat([ms, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# refer to https://github.com/Jiaxuan-Li/MGU-Net
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1) + x
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        padding = (0, 0)
        # reduce dimension
        self.conv_state = BasicConv2d(num_in, self.num_s, kernel_size=(1, 1), padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj = BasicConv2d(num_in, self.num_n, kernel_size=(1, 1), padding=padding)
        self.conv_reproj = BasicConv2d(num_in, self.num_n, kernel_size=(1, 1), padding=padding)
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=(1, 1), padding=padding, stride=(1, 1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out


class MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = BasicConv2d(n_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.ModuleDict({"GCN00": GloRe_Unit(out_channels, out_channels)})

        self.conv1_1 = BasicConv2d(n_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = BasicConv2d(n_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.ModuleDict({"GCN00": GloRe_Unit(out_channels, out_channels)})

        self.conv2_1 = BasicConv2d(n_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = BasicConv2d(n_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.ModuleDict({"GCN00": GloRe_Unit(out_channels, out_channels/2)})

        self.conv3_1 = BasicConv2d(n_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = BasicConv2d(n_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.ModuleDict({"GCN00": GloRe_Unit(out_channels, out_channels/2)})

        self.f1 = BasicConv2d(n_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0.GCN00(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1.GCN00(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2.GCN00(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3 = self.glou3.GCN00(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)


class AMGUnet(nn.Module):
    """ Full assembly of the parts to form the complete network """

    def __init__(self, n_channels, n_classes, bilinear=False, fuse_mode=False,
                 attention=True, multiscale=True, mgb=True):
        super(AMGUnet, self).__init__()
        self.name = 'AMGUnet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse = fuse_mode

        if attention:
            self.inc = DoubleConv(n_channels, 64)
        else:
            self.inc = DoubleConv_noAttention(n_channels, 64)
        self.down1 = Down(64, 128, attention)
        self.down2 = Down(128, 256, attention)
        self.down3 = Down(256, 512, attention)
        self.down4 = Down(512, 1024, attention)

        self.mgb = MGR_Module(1024, 2048) if mgb else lambda x: x

        self.up1 = Up(1024, 512, bilinear, multiscale=multiscale)
        self.up2 = Up(512, 256, bilinear, multiscale=multiscale)
        self.up3 = Up(256, 128, bilinear, multiscale=multiscale)
        self.up4 = Up(128, 64, bilinear, multiscale=multiscale)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # c 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)

        x5 = self.mgb(x5)  # mgb

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = [x2, x3, x4]
        if not self.fuse:
            out = self.outc(x)
        return out
