import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.segmentation import _load_model


class DeepLab(nn.Module):
    def __init__(self, arch_type: str = "deeplabv3",
                        backbone: str = "resnet50",
                        pretrained: bool = False,
                        progress: bool = False,
                        num_input_chans: int = 3,
                        num_classes: int = 21,
                        aux_loss: bool = False,
                        **kwargs
                        ):
        super().__init__()

        self.deeplab = _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs)

        shape = list(self.deeplab.backbone.conv1.weight.shape)
        if shape[1] != num_input_chans:
            shape[1] = num_input_chans
            self.deeplab.backbone.conv1.weight = torch.nn.Parameter(torch.rand(shape, requires_grad=True))
            self.deeplab.backbone.conv1.reset_parameters()

    def forward(self, x):
        return self.deeplab(x)