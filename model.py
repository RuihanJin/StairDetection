import torch
import torch.nn as nn
import torch.nn.functional as F


def createModel(opt):
    if opt.model == 'convnet':
        model = ConvNet()
    elif opt.model == 'vggnet':
        model = VGG11()
    elif opt.model == 'resnet':
        model = ResNet101()
    elif opt.model == 'densenet':
        model = DenseNet()
    return model


class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        hidden_layer = [self.in_channels, 16, 32, 64, 32]
        self.batch_norm = [nn.BatchNorm2d(hidden_layer[i]) for i in range(1, len(hidden_layer))]
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layer_0 = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_layer[1], (5, 5), (2, 2), padding=2),
            self.batch_norm[0],
            self.relu,
            self.pool,
        )
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(hidden_layer[1], hidden_layer[2], (3, 3), (2, 2), padding=1),
            self.batch_norm[1],
            self.relu,
            self.pool,
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(hidden_layer[2], hidden_layer[3], (3, 3), (2, 2), padding=1),
            self.batch_norm[2],
            self.relu,
            self.pool,
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(hidden_layer[3], hidden_layer[4], (3, 3), (1, 1), padding=1),
            self.batch_norm[3],
            self.relu,
        )
        self.flatten = nn.Flatten()
        self.dp1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_layer[-1]*4*4, 64)
        self.fc2 = nn.Linear(64, out_classes)

    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)
        return x


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        from torchvision.models import vgg11
        self.net = vgg11(pretrained=True)
        self.class_num = 2
        self.net.classifier.add_module('add_linear', nn.Linear(1000, 19))
    
    def forward(self, x):
        return self.net(x)


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        from torchvision.models import resnet101
        self.net = resnet101(pretrained=True)
        self.class_num = 2
        self.net.fc = nn.Linear(self.net.fc.in_features, self.class_num)
    
    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        from torchvision.models import densenet121
        self.net = densenet121(pretrained=True)
        self.class_num = 2
        self.net.classifier = nn.Linear(self.net.classifier.in_features, self.class_num)
    
    def forward(self, x):
        return self.net(x)
