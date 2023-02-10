from torchvision.models import resnet50
import torch
model = resnet50(pretrained=True).cuda()
input = torch.randn(1, 3, 224, 224).cuda()
print(model(input).shape)
