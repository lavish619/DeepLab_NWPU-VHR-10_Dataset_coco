import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, atr_sep_conv, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        if atr_sep_conv:
            self.add_module(
                "conv1",
                nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, dilation, bias=False, groups=in_ch),
            )
            self.add_module(
                "conv2",
                nn.Conv2d( in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            )

        else:
            self.add_module(
                "conv",
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False ),
            )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1))

        if relu:
            self.add_module("relu", nn.ReLU())


class _ImagePool(nn.Module):
    """
    Image Pooling Layer in the ASPP module.
    """

    def __init__(self, in_ch, out_ch, atr_sep_conv):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1, atr_sep_conv)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

class _ASPP_V3plus(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates, atr_sep_conv):
        super(_ASPP_V3plus, self).__init__()

        self.add_module("c0",_ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1, atr_sep_conv))

        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i+1), _ConvBnReLU(in_ch, out_ch, 3, 1, rate, rate, atr_sep_conv),
            )
        self.add_module("imagepool", _ImagePool(in_ch, out_ch, atr_sep_conv))


    def forward(self, x):
        return torch.cat([stage(x) for stage in self.children()], dim=1)

class DeepLabV3Plus(nn.Module):
    """
    Deeplabv3+ 
    """
    def __init__(self, n_blocks, n_classes, atrous_rates, multi_grids=None, output_stride=16, atr_sep_conv= True):
        super(DeepLabV3Plus, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        if multi_grids is None:
            multi_grids= [1 for _ in range(n_blocks[3])]
        else:
            assert len(multi_grids) == n_blocks[3]


        resnet = torchvision.models.resnet101(pretrained=True)
        
        self.stem = nn.Sequential(OrderedDict([
                                               ('conv1',resnet.conv1),
                                               ('bn1',resnet.bn1),
                                               ('relu',resnet.relu),
                                               ('maxpool',resnet.maxpool)
                                               ]))
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        save_weight = self.layer3[0].downsample[0].weight
        self.layer3[0].downsample[0] = nn.Conv2d(512,
                                                1024,
                                                kernel_size=1,
                                                stride=s[2], 
                                                bias=False)
        with torch.no_grad():
            self.layer3[0].downsample[0].weight.copy_(save_weight)

        for i in range(n_blocks[2]):
            save_weight =  self.layer3[i].conv2.weight
            self.layer3[i].conv2 = nn.Conv2d(256,
                                            256,
                                            kernel_size=3,
                                            stride=s[2] if i==0 else 1, 
                                            padding=d[2],
                                            dilation=d[2], 
                                            bias=False)
            with torch.no_grad():
                self.layer3[i].conv2.weight.copy_(save_weight)

        save_weight = self.layer4[0].downsample[0].weight
        self.layer4[0].downsample[0] = nn.Conv2d(1024,
                                                2048,
                                                kernel_size=1,
                                                stride=s[3], 
                                                bias=False)
        with torch.no_grad():
            self.layer4[0].downsample[0].weight.copy_(save_weight)

        for i in range(n_blocks[3]):
            save_weight =  self.layer4[i].conv2.weight
            self.layer4[i].conv2 = nn.Conv2d(512,
                                            512,
                                            kernel_size=3,
                                            stride=s[3] if i==0 else 1, 
                                            padding=d[3]*multi_grids[i],
                                            dilation=d[3]*multi_grids[i], 
                                            bias=False)
            with torch.no_grad():
                self.layer4[i].conv2.weight.copy_(save_weight)

        self.assp =  _ASPP_V3plus(2048, 256, atrous_rates, atr_sep_conv)

        concat_ch = 256 * (len(atrous_rates) + 2)

        self.fc1 = _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1, atr_sep_conv)


        # Decoder
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1, atr_sep_conv)

        if atr_sep_conv:
            conv3 = ('conv3', nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, groups=256),
                                            nn.Conv2d(256, n_classes, kernel_size=1)
                                            )
            )
        else:
            conv3 = ("conv3", nn.Conv2d(256, n_classes, kernel_size=1))

        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(304, 256, 3, 1, 1, 1, atr_sep_conv)),
                    ("conv2", _ConvBnReLU(256, 256, 3, 1, 1, 1, atr_sep_conv)),
                    conv3
                ]
            )
        )


    def forward(self, x):
        N,C,H,W = x.size()
        
        x = self.stem(x)
        x = self.layer1(x)
        f0 = self.reduce(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.assp(x)
        x = self.fc1(x)

        x = F.interpolate(x, size=f0.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, f0), dim=1)
        x = self.fc2(x)

        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x 

if __name__ == '__main__':
    model = DeepLabV3Plus(n_classes=21, 
                    n_blocks=[3, 4, 23, 3], 
                    atrous_rates=[6, 12, 18], 
                    multi_grids = [1,2,4],
                    output_stride=16,
                    atr_sep_conv=True)

    image = torch.randn(4, 3, 300, 300)
    print("input:", image.shape)
    print(model(image).shape)

    # print(model)
