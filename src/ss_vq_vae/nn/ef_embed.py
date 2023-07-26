from enum import Enum
import confugue
import torch
from torch import nn
import torchvision.models


class EfficientNetType(Enum):
    EFFICIENTNET_B0 = 0


@confugue.configurable
class EfficientNetEmbedding(nn.Module):
    def __init__(self, backend_type: EfficientNetType, output_size: int):
        super(EfficientNetEmbedding, self).__init__()
        self.backend = EfficientNetEmbedding.get_backend(backend_type)
        self.fc = nn.Identity() if output_size == 1280 else nn.Linear(1280, output_size)

    def forward(self, x: torch.Tensor):
        x = self.backend(x)
        # x, _ = torch.max(x, (1, 2))
        x = self.fc(x)
        return x

    def get_backend(backend_type: EfficientNetType, pretrained: bool = False):
        if backend_type == EfficientNetType.EFFICIENTNET_B0:
            return torchvision.models.efficientnet_b0(pretrained=pretrained, num_classes=1280)
        # TODO
    pass
