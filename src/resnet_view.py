import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=False)
print(resnet18)