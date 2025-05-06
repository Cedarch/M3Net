import torch
from torch import nn


class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict
    
class SE_Blockv2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att_det = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.att_seg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x, det,seg):
        return x * self.att_det(det) + x * self.att_det(seg)
    
class MAFI(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        lidar_channel = self.model_cfg.LIDAR_CHANNEL
        img_channel = self.model_cfg.IMG_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL

        self.conv_det = nn.Sequential(
            nn.Conv2d(lidar_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        self.conv_seg = nn.Sequential(
            nn.Conv2d(img_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )

        self.se_layer = SE_Blockv2(out_channel)


    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
 
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features_lidar']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        lidar_feat = self.conv_det(lidar_bev)
        img_feat = self.conv_seg(img_bev)
        mm_bev = self.conv(cat_bev)
        mm_bev = self.se_layer(mm_bev,lidar_feat,img_feat)       
        batch_dict['spatial_features'] = mm_bev
        return batch_dict