import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import ipdb
st = ipdb.set_trace

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        # self.resnet = resnet

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, images):
        out = self.resnet(images)  # dimension: (batchsize * n frame, 3, 227, 227)
        print("num params",sum([np.prod(p.size()) for p in self.resnet.parameters()]))
        print(out.shape)
        st()
        return out


resnet = ResnetEncoder()
resnet(torch.randn(1,3,227,227))