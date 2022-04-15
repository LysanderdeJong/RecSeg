import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLab(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        progress: bool = False,
        num_input_chans: int = 3,
        num_classes: int = 21,
        aux_loss: bool = False,
        **kwargs
    ):
        super().__init__()

        self.deeplab = deeplabv3_resnet50(
            pretrained, progress, num_classes, aux_loss, **kwargs
        )

        shape = list(self.deeplab.backbone.conv1.weight.shape)
        if shape[1] != num_input_chans:
            shape[1] = num_input_chans
            self.deeplab.backbone.conv1.weight = torch.nn.Parameter(
                torch.rand(shape, requires_grad=True)
            )
            self.deeplab.backbone.conv1.reset_parameters()

    def forward(self, x):
        return self.deeplab(x)
