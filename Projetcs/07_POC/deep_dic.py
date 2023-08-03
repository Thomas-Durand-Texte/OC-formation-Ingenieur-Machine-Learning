""" Code disponible sur https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation/tree/main
"""
# %%
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet
from torch.nn import init
###


# %%
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant_(layer.bias, 0)
    return layer


# Returns 2D batch normalisation layer
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    # Use mean 0, standard deviation 1 init
    init.constant_(layer.weight, 1)
    init.constant_(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 14, 16, 3], 1000)
        self.conv_f = conv(2,64, kernel_size=3,stride = 1)
        self.ReLu_1 = nn.ReLU(inplace=True)
        self.conv_pre = conv(512, 1024, stride=2, transposed=False)
        self.bn_pre = bn(1024)

    def forward(self, x):
        x1 = self.conv_f(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.ReLu_1(self.bn_pre(self.conv_pre(x5)))
        return x1, x2, x3, x4, x5,x6


class SegResNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        #self.conv3 = conv(1024,1024, stride=1, transposed=False)
        #self.bn3 = bn(1024)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512,512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes,stride=2, kernel_size=5)
        init.constant_(self.conv10.weight, 0)  # Zero init

    def forward(self, x):
        
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x+x4 )))
        x = self.relu(self.bn7(self.conv7(x+x3 )))
        x = self.relu(self.bn8(self.conv8(x+x2 )))
        x = self.relu(self.bn9(self.conv9(x+x1 )))
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x / -2.


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 14, 16, 3], 1000)
        self.conv_f = conv(2,64, kernel_size=3,stride = 1)
        self.ReLu_1 = nn.ReLU(inplace=True)
        self.conv_pre = conv(512, 1024, stride=2, transposed=False)
        self.bn_pre = bn(1024)

    def forward(self, x):
        x1 = self.conv_f(x)
        #print('x1',x1.size())
        x = self.bn1(x1)
        #print(x.size())
        x = self.relu(x)
        #print(x.size())
        x2 = self.maxpool(x)
        #print('x2',x2.size())
        x = self.layer1(x2)
        #print(x2.size())
        x3 = self.layer2(x)
        #print('x3',x3.size())
        x4 = self.layer3(x3)
        #print('x4',x4.size())
        x5 = self.layer4(x4)
        #print('x5',x5.size())
        x6 = self.ReLu_1(self.bn_pre(self.conv_pre(x5)))
        #print('x6',x6.size())
        return x1, x2, x3, x4, x5,x6


class SegResNet_strain(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        #self.conv3 = conv(1024,1024, stride=1, transposed=False)
        #self.bn3 = bn(1024)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512,512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=1, transposed=False)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes,stride=2, kernel_size=3)
        init.constant_(self.conv10.weight, 0)  # Zero init

    def forward(self, x):
        #b,c,w,h = x.size()
        #x = x.view(b,c,w,h)
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        #x1 = x1.view(b,x1.size(1),x1.size(2),x1.size(3))
        #x2 = x2.view(b,x2.size(1),x2.size(2),x2.size(3))
        #x3 = x3.view(b,x3.size(1),x3.size(2),x3.size(3))
        #x4 = x4.view(b,x4.size(1),x4.size(2),x4.size(3))
        #x5 = x5.view(b,x5.size(1),x5.size(2),x5.size(3))

        #x1 = torch.max(x1,1)[0]
        #x2 = torch.max(x2,1)[0]
        #x3 = torch.max(x3,1)[0]
        #x4 = torch.max(x4,1)[0]
        #x5 = torch.max(x5,1)[0]


        #print(x5.size())
        #print(x4.size())
        #print(x1.size())
        #x = self.relu(self.bn3(self.conv3(x6)))
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        #print(x.size())
        x = self.relu(self.bn6(self.conv6(x+x4 )))
        #print(x.size())
        x = self.relu(self.bn7(self.conv7(x+x3 )))
        #print(x.size())
        x = self.relu(self.bn8(self.conv8(x+x2 )))
        #print(x.size())
        x = self.relu(self.bn9(self.conv9(x+x1 )))
        #print(x.size())
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x * -0.01


###


# %%
def get_model(filename_data, mode='displacement'):
    fnet = FeatureResNet()
    if mode == 'displacement':
        fcn = SegResNet(2, fnet)
    elif mode == 'strain':
        fcn = SegResNet_strain(4, fnet)
    fcn = fcn.cuda()
    fcn.load_state_dict(torch.load(filename_data))
    return fcn
###


# %% END OF FILE
###
