import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DeepLabLargeFOV(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(DeepLabLargeFOV, self).__init__(*args, **kwargs)
        vgg16 = torchvision.models.vgg16()

        layers = []
        layers.append(nn.Conv2d(in_dim, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))

        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))
        self.features = nn.Sequential(*layers)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride = 1, padding = 1))
        classifier.append(nn.Conv2d(512,
            1024,
            kernel_size = 3,
            stride = 1,
            padding = 12,
            dilation = 12))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Conv2d(1024, out_dim, kernel_size=1))
        self.classifier = nn.Sequential(*classifier)

        self.init_weights()


    def forward(self, x):
        N, C, H, W = x.size()
        x = self.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x

    def init_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_vgg = vgg.features.state_dict()
        self.features.load_state_dict(state_vgg)

        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)

class VGG16_LargeFOV(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG16_LargeFOV, self).__init__()
        self.features = nn.Sequential(
            ### conv1_1 conv1_2 maxpooling
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv2_1 conv2_2 maxpooling
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv3_1 conv3_2 conv3_3 maxpooling
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),


            ### conv4_1 conv4_2 conv4_3 maxpooling(stride=1)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            ### conv5_1 conv5_2 conv5_3 (dilated convolution dilation=2, padding=2)
            ### maxpooling(stride=1)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ### average pooling
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),

            ### fc6 relu6 drop6
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            ### fc7 relu7 drop7 (kernel_size=1, padding=0)
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            ### fc8
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        N,C,H,W = x.size()
        output = self.features(x)
        output = nn.functional.interpolate(output, size=(H,W), mode='bilinear', align_corners=True)
        return output
    
    def _initialize_weights(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    nn.init.normal_(m[1].weight.data, mean=0, std=0.01)
                    nn.init.constant_(m[1].bias.data, 0.0)