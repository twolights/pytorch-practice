import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(device=None, pretrained=False) -> nn.Module:
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None)
    # model.conv1 = nn.Conv2d(depth, 64, kernel_size=3, stride=1, padding=0, bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=2)
    if device is not None:
        model.to(device)
    return model
