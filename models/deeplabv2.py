import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()

        ch = [64 * 2 ** p for p in range(6)]

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
                                                stride=1, 
                                                bias=False)
        with torch.no_grad():
            self.layer3[0].downsample[0].weight.copy_(save_weight)

        for i in range(n_blocks[2]):
            save_weight =  self.layer3[i].conv2.weight
            self.layer3[i].conv2 = nn.Conv2d(256,
                                            256,
                                            kernel_size=3,
                                            stride=1, 
                                            padding=2,
                                            dilation=2, 
                                            bias=False)
            with torch.no_grad():
                self.layer3[i].conv2.weight.copy_(save_weight)

        save_weight = self.layer4[0].downsample[0].weight
        self.layer4[0].downsample[0] = nn.Conv2d(1024,
                                                2048,
                                                kernel_size=1,
                                                stride=1, 
                                                bias=False)
        with torch.no_grad():
            self.layer4[0].downsample[0].weight.copy_(save_weight)

        for i in range(n_blocks[3]):
            save_weight =  self.layer4[i].conv2.weight
            self.layer4[i].conv2 = nn.Conv2d(512,
                                            512,
                                            kernel_size=3,
                                            stride=1, 
                                            padding=4,
                                            dilation=4, 
                                            bias=False)
            with torch.no_grad():
                self.layer4[i].conv2.weight.copy_(save_weight)
        
        self.assp =  _ASPP(ch[5], n_classes, atrous_rates)
       

    def forward(self, x):
        N,C,H,W = x.size()

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x= self.assp(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x 

if __name__ == '__main__':
    model = DeepLabV2(n_classes=21, 
    n_blocks=[3, 4, 23, 3], 
    atrous_rates=[6,12,18,24])

    image = torch.randn(2, 3, 300, 300)
    print("input:", image.shape)
    print("output:", model(image).shape)

    # print(model)
