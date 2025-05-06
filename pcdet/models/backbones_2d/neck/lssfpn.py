import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class LSSFPN(nn.Module):
    def __init__(
        self,
        model_cfg
    ) -> None:
        super().__init__()
        self.in_indices = model_cfg.in_indices
        self.in_channels = model_cfg.in_channels
        self.out_channels = model_cfg.out_channels
        self.scale_factor = model_cfg.scale_factor
        self.fuse = nn.Sequential(
            nn.Conv2d(self.in_channels[0] + self.in_channels[1], self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )
        if self.scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=self.scale_factor,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(True),
            )

    def forward(self, data_dict) -> torch.Tensor:
        x = data_dict['spatial_features_2d']
        x1 = x[self.in_indices[0]]
        assert x1.shape[1] == self.in_channels[0]
        x2 = x[self.in_indices[1]]
        assert x2.shape[1] == self.in_channels[1]

        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x1, x2], dim=1)

        x = self.fuse(x)
        if self.scale_factor > 1:
            x = self.upsample(x)
        data_dict['spatial_features_2d'] = x
        x = self.upsample(x)
        data_dict['spatial_features_2d_4x'] = x
        return data_dict
