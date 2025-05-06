import torch
import os
import cv2
import copy
from torch import nn
from torch.nn import functional as F
from ..model_utils.transfusion_utils import clip_sigmoid
from ...utils import loss_utils
import numpy as np
from ..model_utils import model_nms_utils
from collections import OrderedDict
from torch.nn.init import kaiming_normal_
from ...utils.spconv_utils import replace_feature, spconv
from .target_assigner.hungarian_assigner import HungarianAssigner3D
try:
    from tools.visual_utils import open3d_vis_utils as V
except:
    pass
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.m3net_utils import  PositionEmbeddingLearned, \
       BevFeatureSlicer, FFNLayer, TransformerDecoderMultiTask, \
       MSDeformAttnPixelDecoderEncOnly, TransformerDecoderDet, MLP, SelfAttentionLayer
from ..model_utils import centernet_utils
from ...utils import box_utils, common_utils
from ..model_utils.deform3d_utils import SelfAttentionLayer3D, CE_ssc_loss, sem_scal_loss, geo_scal_loss
from ..model_utils.lovasz_softmax import lovasz_softmax
from ..model_utils.vmamba import VSSM, VSSMENC, VSSBlock, VSSBlock2D
nusc_class_frequencies = np.array([2242961742295, 25985376, 1561108, 28862014, 196106643, 15920504,
                2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
                1446141335, 1724391378])

color_map = {
    0: (0,0,0),
    1: (255,255,255),
    2: (255,0,0),
    3: (0,255,0),
    4: (0,0,255),
    5: (255,255,0),
    6: (0,255,255),
    7: (255,0,255),
    8: (192,192,192),
    9: (128,128,128),
    10: (128,0,0),
    11: (128,128,0),
    12: (0,128,0),
    13: (128,0,128),
    14: (0,128,128),
    15: (0,0,128),
    16: (128,128,128),
    255:(0,0,0)
 }


class SeparateHead_Sparse(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features

        return ret_dict


class SElayerShareGate(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc(y).view(b, c, 1, 1)
        y2 = 1 - y1
        return x * y1.expand_as(x), x * y2.expand_as(x)
    
class SElayerGate(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc(y).view(b, c, 1, 1)
        return x * y1.expand_as(x)


class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    # nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x, mapfeat=None):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            if cur_name=='heatmap' and mapfeat is not None:
                x = x + mapfeat
            if cur_name=='sparse_heatmap':
                continue
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict



class M3NetHead(nn.Module):

    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,receptive_field=None
    ):
        super().__init__()
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class
        self.with_road = model_cfg.SEM_SEG_HEAD.WITH_ROAD
        self.with_line = model_cfg.SEM_SEG_HEAD.WITH_LINE
        self.one_channel = model_cfg.SEM_SEG_HEAD.ONE_CHANNEL
        self.down_sample = model_cfg.SEM_SEG_HEAD.DOWN_SAMPLE
        self.num_proposals_carped = model_cfg.get('NUM_PROPOSALS_CARPED',200)
        self.num_proposals_other = model_cfg.get('NUM_PROPOSALS_OTHER',0)
        self.num_proposals = self.num_proposals_carped + self.num_proposals_other
        self.nms_kernel_size = model_cfg.SEM_SEG_HEAD.NMS_KERNEL_SIZE
        hidden_channel = model_cfg.SEM_SEG_HEAD.HIDDEN_CHANNEL
        decoder_channel = model_cfg.MASK_FORMER.HIDDEN_DIM
        self.hidden_channel = hidden_channel
        query_channel = model_cfg.QUERY_CHANNEL
        self.decoder_channel = decoder_channel
        self.query_channel = query_channel
        self.model_cfg = model_cfg

        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=10, num_conv=self.model_cfg.NUM_HM_CONV)

        self.prediction_head = SeparateHead_Transfusion(query_channel, 64, 1, heads, use_bias=bias)

        if self.model_cfg.TRAIN_MAP:
            num_seg_query = self.model_cfg.get('SEG_QUERY_NUM',6)
            self.query_feat = nn.Embedding(num_seg_query,self.query_channel)
            self.query_embed = nn.Embedding(num_seg_query,self.query_channel)
        else:
            self.query_feat = nn.Embedding(20,self.query_channel)
            self.query_embed = nn.Embedding(20,self.query_channel)

        input_shape  = {'spatial_features_2d':{"channel":512,"stride":8}}

        self.pixel_decoder,self.pixel_decoder2 = self.pixel_decoder_init(model_cfg, input_shape)
        self.predictor,self.predictor_line = self.predictor_init(model_cfg)

        self.shared_conv = nn.Conv2d(in_channels=512,out_channels=query_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(query_channel,query_channel, kernel_size=3,padding=1,bias=False))
        layers.append(nn.Conv2d(in_channels=query_channel,out_channels=10,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        layers = []
        layers.append(BasicBlock2D(query_channel,query_channel, kernel_size=3,padding=1,bias=False))
        layers.append(nn.Conv2d(in_channels=query_channel,out_channels=6,kernel_size=3,padding=1))
        self.segmap_head = nn.Sequential(*layers)

        self.seg_map_attn = SelfAttentionLayer(                    
                                                    d_model=self.query_channel,
                                                    nhead=8,
                                                    dropout=0.0,
                                                    normalize_before=False,
                                                    )

        self.dist_map_attn = SelfAttentionLayer(                    
                                                    d_model=self.query_channel,
                                                    nhead=8,
                                                    dropout=0.0,
                                                    normalize_before=False,
                                                    )

        self.class_encoding = nn.Conv1d(10, query_channel, 1)
        self.pos_layer_query = PositionEmbeddingLearned(2, query_channel)
        self.pos_layer_key = PositionEmbeddingLearned(2, hidden_channel)
        self.pos_layer_key = PositionEmbeddingLearned(2, hidden_channel)
        # self.pos_layer_key_dec = PositionEmbeddingLearned(2, decoder_channel)
        if self.model_cfg.get('TRAIN_OCC',False):
            self.pos_layer_occ = PositionEmbeddingLearned(3, query_channel)
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.feature_map_stride  =1

        self.grid_conf = {
            'xbound': [-54, 54, 0.6],
            'ybound': [-54, 54, 0.6],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.map_grid_conf = {
            'ybound': [-30.0, 30.0, 0.15],
            'xbound': [-15.0, 15.0, 0.15],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.motion_grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.occ_grid_conf = {
            'xbound': [-51.2, 51.2, 0.6],
            'ybound': [-51.2, 51.2, 0.6],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.cubeslicer = BevFeatureSlicer(grid_conf=self.grid_conf,map_grid_conf=self.occ_grid_conf)
        self.maskslicer = BevFeatureSlicer(grid_conf=self.grid_conf,map_grid_conf=self.motion_grid_conf)
        self.mapslicer = BevFeatureSlicer(grid_conf=self.grid_conf,map_grid_conf=self.map_grid_conf)
        self.reverseslicer = BevFeatureSlicer(grid_conf=self.motion_grid_conf,map_grid_conf=self.grid_conf)
        self.voxel_size = [0.6,0.6,20]
        self.bev_pos = self.create_2D_grid(180, 180)
        self.occ_pos = self.create_3D_grid(180, 180, 5)
        self.bev_map_pos = self.create_2D_grid(200, 200)
        self.code_size = 10
        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.map_cls = loss_utils.SigmoidFocalClassificationLossV2(gamma=2.0,alpha=-1)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        self.load_line = False
        self.fuser_loaded = False
        self.center_norm = nn.LayerNorm(self.query_channel)
        self.center_map_norm = nn.LayerNorm(self.query_channel)
        self.center_mlp = MLP(self.query_channel, self.query_channel, self.query_channel, 3)
        self.center_query = nn.Embedding(10, self.query_channel)
        self.center_embed = nn.Embedding(10, self.query_channel)
        self.querymap_cls_feat = nn.Embedding(6, self.query_channel)

        if self.model_cfg.get('TRAIN_OCC',False):
            self.occ_ffn = nn.ModuleList()
            self.occ_attn = nn.ModuleList()
            for _ in range(2):
                self.occ_attn.append(SelfAttentionLayer3D(
                                                        d_model=self.hidden_channel,
                                                        nhead=8,
                                                        dropout=0.0,
                                                        normalize_before=False,
                                                        bev_z= 11 if self.model_cfg.get('USE_11_HIGHT',False) else 5 ,
                                                        ))
                self.occ_ffn.append(
                        FFNLayer(
                            d_model=self.hidden_channel,
                            dim_feedforward=1024,
                            dropout=0.0,
                            normalize_before=False,
                        ))
            self.mask_embed = nn.Embedding(1, self.query_channel)

            self.voxel_mlp_head = MLP(self.query_channel,self.query_channel,17,3)

            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
            self.empty_idx = 0

        if self.model_cfg.get('TRAIN_OCC',False):
            self.loss_weight_cfg = self.model_cfg.loss_weight_cfg
            self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
            self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
            self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
            self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
            if self.model_cfg.get('USE_MAMBA3D_ATTN',False):
                self.mamba_hw = nn.ModuleList()
                self.mamba_z = nn.ModuleList()
                for _ in range(2):
                    self.mamba_hw.append(
                        VSSBlock(
                            hidden_dim=self.hidden_channel,
                            mlp_ratio=0.0,
                            d_state=self.model_cfg.get('D_STATE',8),
                            ssm_ratio=1.0
                        ))

                    self.mamba_z.append(
                        VSSBlock2D(
                        hidden_dim=self.hidden_channel,
                        mlp_ratio=0.0,
                        d_state=self.model_cfg.get('D_STATE',8),
                        ssm_ratio=1.0
                    ))

            if self.model_cfg.get('USE_MAMBA3D_ATTN_V2',False):
                self.mamba_h = nn.ModuleList()
                self.mamba_w = nn.ModuleList()
                self.mamba_z = nn.ModuleList()
                for _ in range(2):
                    self.mamba_h.append(
                        VSSBlock2D(
                            hidden_dim=self.hidden_channel,
                            mlp_ratio=0.0,
                            d_state=8,
                            ssm_ratio=1.0
                        ))

                    self.mamba_w.append(
                        VSSBlock2D(
                            hidden_dim=self.hidden_channel,
                            mlp_ratio=0.0,
                            d_state=8,
                            ssm_ratio=1.0
                        ))

                    self.mamba_z.append(
                        VSSBlock2D(
                        hidden_dim=self.hidden_channel,
                        mlp_ratio=0.0,
                        d_state=8,
                        ssm_ratio=1.0
                    ))

        self.gate_heat = MLP(self.query_channel, self.query_channel, self.query_channel, 3)
        self.gate_map = MLP(self.query_channel, self.query_channel, self.query_channel, 3)

        self.map_norm_enc = nn.LayerNorm(self.query_channel)
        self.map_embed_enc = MLP(self.query_channel, self.query_channel, self.query_channel, 3)
        self.map_norm_dec = nn.LayerNorm(self.query_channel)
        self.map_embed_dec = MLP(self.query_channel, self.query_channel, self.query_channel, 3)


        self.only_train_box = self.model_cfg.TRAIN_BOX and (not self.model_cfg.TRAIN_MAP) and (not self.model_cfg.TRAIN_MASK) and (not self.model_cfg.TRAIN_OCC)
        self.only_train_occ = self.model_cfg.TRAIN_OCC and (not self.model_cfg.TRAIN_MAP) and (not self.model_cfg.TRAIN_MASK) and (not self.model_cfg.TRAIN_BOX)

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MASK_FORMER.NHEADS
        transformer_dim_feedforward = cfg.MASK_FORMER.DIM_FEEDFORWARD
        transformer_enc_layers = cfg.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features =  cfg.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES # ["res3", "res4", "res5"]

        if self.model_cfg.get('USE_VM_ENC', False):
    
            pixel_decoder = VSSMENC(in_chans=512,
                                depths = [self.model_cfg.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS],
                                ssm_ratio= self.model_cfg.get('SSM_RATIO',2),
                                mlp_ratio=self.model_cfg.get('VM_MLP',0.0),
                                d_state = self.model_cfg.get('D_STATE'),
                                config=self.model_cfg,
                                dims=[self.hidden_channel])
                    

        else:
            pixel_decoder = MSDeformAttnPixelDecoderEncOnly(input_shape,
                                                    transformer_dropout,
                                                    transformer_nheads,
                                                    transformer_dim_feedforward,
                                                    transformer_enc_layers,
                                                    conv_dim,
                                                    mask_dim,
                                                    transformer_in_features,
                                                    common_stride,
                                                    cfg)

            return pixel_decoder, None


    def predictor_init(self, cfg):

        in_channels = self.query_channel
        mask_score = cfg.MASK_FORMER.ATTN_MASK_SCORE
        hidden_dim = cfg.MASK_FORMER.HIDDEN_DIM
        self.num_queries = self.num_proposals 
        nheads = cfg.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MASK_FORMER.PRE_NORM
        mask_dim = cfg.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        mask_classification = True

        if self.model_cfg.TRAIN_MAP:
            predictor = TransformerDecoderMultiTask(in_channels, 
                                                                self.num_classes, 
                                                                mask_classification,
                                                                hidden_dim,
                                                                self.num_queries,
                                                                nheads,
                                                                dim_feedforward,
                                                                dec_layers,
                                                                pre_norm,
                                                                mask_dim,
                                                                enforce_input_project,
                                                                mask_score,
                                                                self.down_sample,
                                                                cfg)
            predictor_line=None
        else:

            predictor = TransformerDecoderDet(in_channels, 
                                                                self.num_classes, 
                                                                mask_classification,
                                                                hidden_dim,
                                                                self.num_queries,
                                                                nheads,
                                                                dim_feedforward,
                                                                dec_layers,
                                                                pre_norm,
                                                                mask_dim,
                                                                enforce_input_project,
                                                                mask_score,
                                                                self.down_sample,
                                                                cfg)
            predictor_line=None
        return predictor,predictor_line

    def losses_init(self, cfg):

        try:
            local_rank=int(os.environ['LOCAL_RANK'])
        except:
            local_rank= 0
        self.device = torch.device("cuda", local_rank)
        self.map_weight = cfg.MASK_FORMER.MAP_WEIGHT
        self.det_weight = cfg.MASK_FORMER.DET_WEIGHT
        self.heatmap_weight = cfg.MASK_FORMER.HEATMAP_WEIGHT
        self.map_enc_weight = cfg.MASK_FORMER.MAP_ENC_WEIGHT

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def create_3D_grid(self, x_size, y_size, z_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size], [0, z_size - 1, z_size]]
        # NOTE: modified
        batch_x, batch_y, batch_z = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        batch_z = batch_z + 0.5
        coord_base_3d = torch.cat([batch_x[None], batch_y[None],batch_z[None]], dim=0)[None]
        coord_base = coord_base_3d.view(1, 3, -1).permute(0, 2, 1)
        return coord_base

    def get_bev_pos_embed(self,inputs):
        batch_size = inputs.shape[0]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(inputs.device)
        bev_map_pos = self.bev_map_pos.repeat(batch_size, 1, 1).to(inputs.device)
        # bev_pos_4x = self.bev_pos_4x.repeat(batch_size, 1, 1).to(inputs.device)
        if self.decoder_channel != self.hidden_channel:
            bev_embed_dec = self.pos_layer_key_dec(bev_pos.flip(dims=[-1])).reshape(batch_size,self.query_channel,inputs.shape[-2],inputs.shape[-1])
        else:
            bev_embed_dec =None
        bev_embed = self.pos_layer_key(bev_pos.flip(dims=[-1])).reshape(batch_size,self.hidden_channel,inputs.shape[-2],inputs.shape[-1])
        bev_map_embed = None #self.pos_layer_key(bev_map_pos.flip(dims=[-1])).reshape(batch_size,self.hidden_channel,200,200)
        return bev_pos, bev_embed, bev_map_embed,bev_embed_dec

    def get_occ_pos_embed(self,inputs,bev_z):
        batch_size = inputs.shape[0]
        occ_pos = self.occ_pos.repeat(batch_size, 1, 1).to(inputs.device)
        occ_embed = self.pos_layer_occ(occ_pos[...,[1,0,2]]).reshape(batch_size,self.hidden_channel,inputs.shape[-2],inputs.shape[-1],bev_z)
        return occ_pos, occ_pos.reshape(1,180,180,bev_z,3),occ_embed

    def get_occ_query_feat(self, inputs,top_proposals_index):
        
        batch_size = inputs.shape[0]
        lidar_feat =  self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        top_proposals_index = top_proposals_index[:,0:1] + top_proposals_index[:,1:2]*180
        query_feat = lidar_feat_flatten.gather(index=top_proposals_index.long().transpose(0,1)[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),dim=-1)
        return query_feat, top_proposals_index


    def get_query_embed_enc_only(self, batch_dict, inputs, bev_pos, bev_embed):
        batch_size = inputs.shape[0]
        lidar_feat =  self.shared_conv(inputs)
        if self.model_cfg.get('USE_MAMBA_PROJ_EMBED',False):
            if self.model_cfg.get('SHARE_MAP_HEAT_EMBED',False):
                heat_features = self.gate_map(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
                map_features = self.maskslicer(heat_features).transpose(2,3)
                # heat_features = self.gate_heat(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
            else:
                map_features = self.gate_map(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
                map_features = self.maskslicer(map_features).transpose(2,3)
                heat_features = self.gate_heat(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
        else:
            if self.model_cfg.get('BEV_USE_GATE',False):
                map_features = self.gate_map(lidar_feat)
                map_features = self.maskslicer(map_features).transpose(2,3)
                heat_features = self.gate_heat(lidar_feat)
            else:
                map_features = self.gate_map(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
                map_features = self.maskslicer(map_features).transpose(2,3)
                heat_features = self.gate_heat(lidar_feat.permute(0,2,3,1)).permute(0,3,1,2)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)


        # center_map_embed = torch.cat([self.center_embed.weight,self.query_embed.weight],0)[None].repeat(batch_size,1,1).permute(1,0,2)
        center_map_query = torch.cat([self.center_query.weight,self.query_feat.weight],0)[None].repeat(batch_size,1,1).permute(1,0,2)

        if self.model_cfg.get('CENTER_MAP_ATTN',False):
            center_map_query = self.center_map_attn(center_map_query,query_pos=None)

        center_query = center_map_query[:10].transpose(0,1)
        query_feat_map = center_map_query[10:].transpose(0,1)


        center_query = self.center_norm(center_query)
        center_query = self.center_mlp(center_query)
        query_feat_map = self.map_norm_enc(query_feat_map)
        query_feat_map = self.map_embed_enc(query_feat_map.transpose(0,1)).transpose(0,1)

        self.mask_features = map_features
        num_seg_query = self.model_cfg.SEG_QUERY_NUM 

        dense_heatmap = self.heatmap_head(lidar_feat)

        if self.model_cfg.get('DEEP_ENC_LOSS',False):
            outputs_line = torch.einsum("bqc,bchw->bqhw", query_feat_map.repeat(3,1,1), torch.cat(map_features_list)).contiguous()
        else:
            if self.model_cfg.get('USE_TOP_SEG',False):
                outputs_line = None
            else:
                if num_seg_query==6:
                    outputs_line = torch.einsum("bqc,bchw->bqhw", query_feat_map, map_features).contiguous()
                else:
                    split = num_seg_query//6
                    b,c,h,w = map_features.shape
                    line_feats = map_features#.reshape(b,c,split,h//split,w)
                    line = query_feat_map.reshape(split,6,query_feat_map.shape[-1])
                    outputs_line_list = []
                    for i in range(split):
                        outputs_line = torch.einsum("bqc,bchw->bqhw", line[i:i+1], line_feats)
                        outputs_line = outputs_line.reshape(batch_size,6,split,h//split,w)
                        outputs_line_list.append(outputs_line[:,:,i])
                    outputs_line = torch.cat(outputs_line_list,2)

        heatmap = dense_heatmap[-1:].detach().sigmoid()
        

        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
        local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1).contiguous()
        
        if self.num_proposals_other > 0:
            heatmap_carped = heatmap.clone()
            heatmap_other = heatmap.clone()
            heatmap_carped[:,[1,2,3,4,5,6,7,9]] = 0
            heatmap_other[:,[0,8]] = 0
            top_proposals_carped = heatmap_carped.view(batch_size, -1).argsort(dim=-1, descending=True)[
                ..., : self.num_proposals_carped
            ]
            top_proposals_other = heatmap_other.view(batch_size, -1).argsort(dim=-1, descending=True)[
                ..., : self.num_proposals_other
            ]
            top_proposals = torch.cat([top_proposals_carped, top_proposals_other], dim=-1)
        else:
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
                ..., : self.num_proposals
            ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]


        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        query_labels = top_proposals_class
        self.query_labels = top_proposals_class
        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=10).permute(0, 2, 1)

        if self.model_cfg.get('QUERY_HEATMAP',False):
            query_cat_encoding = center_query[0].permute(1,0).gather(index=top_proposals_class.expand(self.query_channel,-1),dim=1)[None]
        else:
            query_cat_encoding = self.class_encoding(one_hot.float())

        query_feat += query_cat_encoding
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        feats_with_cls_embed = None
        bg_index = None
        top_seg_index = None
        if self.model_cfg.get('USE_TOP_SEG',False):

            if self.model_cfg.get('MAP_INDEX_AT_BEVSIZE',False):
                map_lidarfeat = lidar_feat
                map_bev_pos = bev_pos.reshape(1,180,180,2)
                # import pdb;pdb.set_trace()
                if self.model_cfg.get('QUERY_SEGMAP',False):
                    querymap_cls_feat = self.querymap_cls_feat.weight[None]
                    pred_map = torch.einsum("bqc,bchw->bqhw", querymap_cls_feat, map_features).contiguous()
                else:
                    pred_map = self.segmap_head(map_lidarfeat)
                map_flatten = pred_map.flatten(2)
                pred_map_bev = self.reverseslicer(pred_map).transpose(2,3).flatten(2)
                pred_map_bev[:,:,top_proposals_index.reshape(-1)] = -1e10
                pred_map_bev = pred_map_bev.reshape(1,6,180,180)
                # split = self.model_cfg.SEG_QUERY_NUM // 6
                split_x = self.model_cfg.SEG_QUERY_NUM // 6
                split_y = 1
                split_map = pred_map_bev.reshape(batch_size, 6, split_x,pred_map_bev.shape[2]//split_x, split_y, pred_map_bev.shape[3]//split_y).permute(0,2,4,1,3,5).contiguous()
                top_seg_index = split_map.reshape(batch_size,split_y*split_x,6,-1).max(dim=-1)[1]
                map_lidarfeat = map_lidarfeat.reshape(batch_size, map_lidarfeat.shape[1], split_x,pred_map_bev.shape[2]//split_x, split_y, pred_map_bev.shape[3]//split_y).permute(0,1,2,4,3,5).contiguous()
                map_lidarfeat = map_lidarfeat.reshape(batch_size, map_lidarfeat.shape[1], split_y*split_x, -1)
                map_bev_pos = map_bev_pos.reshape(batch_size, 2, split_x,pred_map_bev.shape[2]//split_x, split_y, pred_map_bev.shape[3]//split_y).permute(0,1,2,4,3,5).contiguous()
                map_bev_pos = map_bev_pos.reshape(batch_size, 2, split_y*split_x, -1)

                seg_query_pos = map_bev_pos.gather(index=top_seg_index[:, None].expand(-1, 2,-1,-1),dim=-1).reshape(batch_size,2,-1).permute(0,2,1)
                seg_query_pos = seg_query_pos.flip(dims=[-1])
                seg_query_embed = self.pos_layer_query(seg_query_pos).reshape(batch_size,self.query_channel,self.model_cfg.SEG_QUERY_NUM)

                feats_with_cls_embed = torch.zeros(1,lidar_feat_flatten.shape[1],180,180).cuda().flatten(2)
                mask = torch.ones([180*180]).cuda()
                mask[top_proposals_index.reshape(-1).unique()] = 0
                mask[top_seg_index.reshape(-1).unique()] = 0
                bg_index = mask.nonzero().reshape(-1)
                feats_with_cls_embed[:,:,top_proposals_index.reshape(-1)] = query_feat
                feats_with_cls_embed[:,:,top_seg_index.reshape(-1)] = query_feat_map.transpose(1,2)
                feats_with_cls_embed[:,:,bg_index] = lidar_feat_flatten[:,:,bg_index]
                feats_with_cls_embed = feats_with_cls_embed.reshape(1,-1,180,180)

            else:

                map_lidarfeat = self.maskslicer(lidar_feat).transpose(2,3)
                map_bev_pos = self.maskslicer(bev_pos.reshape(1,180,180,2).permute(0,3,1,2)).transpose(2,3)
                # import pdb;pdb.set_trace()
                if self.model_cfg.get('QUERY_SEGMAP',False):
                    querymap_cls_feat = self.querymap_cls_feat.weight[None]
                    pred_map = torch.einsum("bqc,bchw->bqhw", querymap_cls_feat, map_features).contiguous()
                    if 'bev_seg' in batch_dict:
                        batch_dict['bev_seg_pred'] = torch.einsum("bqc,bchw->bqhw", querymap_cls_feat.detach(), batch_dict['bev_seg']).contiguous()
                else:
                    pred_map = self.segmap_head(map_lidarfeat)
                map_flatten = pred_map.flatten(2)
                pred_map2 = pred_map.clone()
                pred_map2 = pred_map2.flatten(2)
                pred_map2[:,:,top_proposals_index.reshape(-1)] = -1e10
                pred_map2 = pred_map2.reshape(1,-1,200,200)
                # split = self.model_cfg.SEG_QUERY_NUM // 6
                split_x = self.model_cfg.SEG_QUERY_NUM // 6
                split_y = 1
                split_map = pred_map2.reshape(batch_size, 6, split_x,pred_map2.shape[2]//split_x, split_y, pred_map2.shape[3]//split_y).permute(0,2,4,1,3,5).contiguous()
                top_seg_index = split_map.reshape(batch_size,split_y*split_x,6,-1).max(dim=-1)[1]
                map_lidarfeat = map_lidarfeat.reshape(batch_size, map_lidarfeat.shape[1], split_x,pred_map2.shape[2]//split_x, split_y, pred_map2.shape[3]//split_y).permute(0,1,2,4,3,5).contiguous()
                map_lidarfeat = map_lidarfeat.reshape(batch_size, map_lidarfeat.shape[1], split_y*split_x, -1)
                map_bev_pos = map_bev_pos.reshape(batch_size, 2, split_x,pred_map2.shape[2]//split_x, split_y, pred_map2.shape[3]//split_y).permute(0,1,2,4,3,5).contiguous()
                map_bev_pos = map_bev_pos.reshape(batch_size, 2, split_y*split_x, -1)

                seg_query_pos = map_bev_pos.gather(index=top_seg_index[:, None].expand(-1, 2,-1,-1),dim=-1).reshape(batch_size,2,-1).permute(0,2,1)
                seg_query_pos = seg_query_pos.flip(dims=[-1])
                seg_query_embed = self.pos_layer_query(seg_query_pos).reshape(batch_size,self.query_channel,self.model_cfg.SEG_QUERY_NUM)

            if self.model_cfg.get('MAP_QUERY_ATTN',False) and self.model_cfg.SEG_QUERY_NUM >6:
                #[1,256,4,6]
                query_feat_map = map_lidarfeat.gather(index=top_seg_index[:, None].expand(-1, map_lidarfeat.shape[1], -1,-1),dim=-1)
                query_feat_map = self.seg_map_attn(query_feat_map[0].permute(1,2,0),query_pos=seg_query_embed.reshape(self.query_channel,split_x,6).permute(1,2,0)).reshape(1,-1,self.query_channel)
                if self.model_cfg.get('DIST_QUERY_ATTN',False) and self.model_cfg.SEG_QUERY_NUM >6:
                    query_feat_map = self.dist_map_attn(query_feat_map[0].reshape(split_x,6,self.query_channel).transpose(0,1),
                                        query_pos=seg_query_embed.reshape(self.query_channel,split_x,6).permute(2,1,0)).transpose(0,1).reshape(1,-1,self.query_channel)
            else:
                query_feat_map = map_lidarfeat.gather(index=top_seg_index[:, None].expand(-1, map_lidarfeat.shape[1], -1,-1),dim=-1).flatten(2).transpose(1,2)
            
            if self.model_cfg.get('QUERY_SEGMAP',False):
                query_feat_map = query_feat_map + querymap_cls_feat[:,None].repeat(1,split_x,1,1).reshape(batch_size,-1,self.query_channel)

        else:
            pred_map = None

        # convert to xy
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])
        query_embed = self.pos_layer_query(query_pos).reshape(batch_size,self.query_channel,self.num_proposals)

        query_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, heatmap.shape[1], -1),
            dim=-1,
        )
        if self.with_road or self.with_line:
            if self.model_cfg.get('USE_TOP_SEG',False):
                if self.model_cfg.get('DET_CLS_ADD_MAP',False):
                    self.det_map_cls = map_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, map_flatten.shape[1], -1),dim=-1,).max(1)[1][:,:,None,None].repeat(1,1,split_x,1)
                    self.det_map_dist = torch.cdist(query_pos,seg_query_pos).reshape(1,200,split_x,-1)
                query_feat_line = query_feat_map.transpose(1,2)
                query_embed_line = seg_query_embed
            else:
                query_feat_line = query_feat_map.repeat(batch_size,1,1).permute(0,2,1)
                query_embed_line = self.query_embed.weight[None].repeat(batch_size,1,1).permute(0,2,1)

            query_feat = torch.cat([query_feat,query_feat_line],-1)
            query_embed = torch.cat([query_embed,query_embed_line],-1)
            query_score = torch.cat([query_score,torch.ones_like(query_score)[...,:query_feat_line.shape[-1]]],-1)
            query_labels = torch.cat([query_labels,torch.ones_like(query_labels)[...,:query_feat_line.shape[-1]]*11],-1)
            if self.model_cfg.get('USE_TOP_SEG',False):
                query_pos = torch.cat([query_pos,seg_query_pos],1)
            else:
                query_pos = torch.cat([query_pos,torch.ones_like(query_pos)[:,:query_feat_line.shape[-1]]*90],1)

        return query_feat, heatmap, query_embed, query_labels, query_pos, bev_pos, dense_heatmap,top_proposals_index, query_score,pred_map, map_features,outputs_line, \
               feats_with_cls_embed, top_seg_index, bg_index

    
    def get_pred_dict_dec(self,predictions,query_pos,heatmap,top_proposals_index,dense_heatmap,inter_references=None,det_map_feat=None):
        
        query_feat_list = predictions['query_feat']
        if self.model_cfg.get('DET_CLS_ADD_MAP',False):
            det_map_dist_index = self.det_map_dist.gather(index=self.det_map_cls,dim=-1).min(2)[1].view(-1)
            det_map_cls_index = self.det_map_cls[0,:,0,0]
            mapfeat = predictions['query_feat'][-1][:,:,200:].reshape(-1,6,self.query_channel)
            batch_index = torch.arange(200).cuda().view(-1)
            det_mapfeat = mapfeat[det_map_dist_index,det_map_cls_index][None].transpose(1,2)
        else:
            det_mapfeat = None
        num_layers = len(query_feat_list)
        bs = query_feat_list[0].shape[0]

        if self.training:
            if self.model_cfg.get('USE_LOCAL_DEC',False):
                res_layer = {}
                res_layer['heatmap'] = []
                res_layer['outputs_coords'] = []
                for idx, query_feat in enumerate(query_feat_list):
                    # res_layer = self.prediction_head_dec[k](query_feat)
                    # res_layer_list.append(res_layer)
                    if idx == 0:
                        reference = self.init_reference # has been sigmoided
                    else:
                        reference = inter_references[idx - 1] # has been sigmoided
                    outputs_class, outputs_coord = self.prediction_head_dec(query_feat[...,:self.num_proposals].permute(0,2,1), reference, idx)
                    res_layer['heatmap'].append(outputs_class[:,None])
                    res_layer['outputs_coords'].append(outputs_coord[:,None])
                res_layer['heatmap'] = torch.cat(res_layer['heatmap'],1).permute(0,1,3,2)
                res_layer['outputs_coords'] = torch.cat(res_layer['outputs_coords'],1)
            else:
                query_feat = torch.cat([feat[...,:self.num_proposals].unsqueeze(1) for feat in query_feat_list],1).reshape(-1,self.query_channel,self.num_proposals)
                res_layer = self.prediction_head(query_feat,det_mapfeat)
                res_layer["center"] = res_layer["center"] + query_pos[:,:self.num_proposals].permute(0, 2, 1)[:,None].repeat(1,res_layer["center"].shape[0]//query_pos.shape[0],1,1).reshape(-1,2,self.num_proposals)

                for key in res_layer.keys():
                    res_layer[key] = res_layer[key].reshape(bs,num_layers,res_layer[key].shape[1],res_layer[key].shape[2])

                if self.model_cfg.get('USE_AUG_LOSS',False):
                    query_feat_list_aux = predictions['query_feat_aux']
                    query_feat_aux = torch.cat([feat.unsqueeze(1) for feat in query_feat_list_aux],1).reshape(-1,self.query_channel,self.num_proposals)
                    res_layer_aux = self.prediction_head(query_feat_aux)
                    res_layer_aux["center"] = res_layer_aux["center"] + query_pos[:,:self.num_proposals].permute(0, 2, 1)[:,None].repeat(1,res_layer_aux["center"].shape[0]//query_pos.shape[0],1,1).reshape(-1,2,self.num_proposals)

                    for key in res_layer_aux.keys():
                        res_layer_aux[key] = res_layer_aux[key].reshape(bs,num_layers,res_layer_aux[key].shape[1],res_layer_aux[key].shape[2])

                    res_layer_aux["query_heatmap_score"] = heatmap.gather(
                        index=top_proposals_index[:, None, :].expand(-1, heatmap.shape[1], -1),
                        dim=-1,
                    )
                    res_layer_aux["dense_heatmap"] = dense_heatmap


                else:
                    res_layer_aux = None


        else:
            query_feat = query_feat_list[-1][...,:self.num_proposals]
            res_layer = self.prediction_head(query_feat,det_mapfeat)
            res_layer["center"] = res_layer["center"] + query_pos[:,:self.num_proposals].permute(0, 2, 1)
            # import pdb;pdb.set_trace()
            if self.model_cfg.get('USE_AUG_LOSS',False):
                query_feat_list_aux = predictions['query_feat_aux']
                query_feat_aux = query_feat_list_aux[-1]
                res_layer_aux = self.prediction_head(query_feat_aux)
                res_layer_aux["center"] = res_layer_aux["center"] + query_pos[:,:self.num_proposals].permute(0, 2, 1)[:,None].repeat(1,res_layer_aux["center"].shape[0]//query_pos.shape[0],1,1).reshape(-1,2,self.num_proposals)
                res_layer_aux["query_heatmap_score"] = heatmap.gather(
                    index=top_proposals_index[:, None, :].expand(-1, heatmap.shape[1], -1),
                    dim=-1,
                )
                res_layer_aux["dense_heatmap"] = dense_heatmap

            else:
                res_layer_aux = None


        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, heatmap.shape[1], -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap

        return res_layer, res_layer_aux

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.dense().reshape(-1,180,180).sum(0).nonzero()
        x_indices = x.indices # index feat use voxel_indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = x_indices[:, 0] 

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(x_indices[batch_inds][:, [3, 2]]) # this is align to gt_box_center
            num_voxels.append(batch_inds.sum())

        return spatial_shape[1:], batch_index, voxel_indices.long(), voxel_indices[:,[1,0]].long(), voxel_indices.shape[0]

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def get_loss_enc_only(self, batch_dict, outputs, gt_labels_det, gt_boxes_det, dense_heatmap,
                         pred_dicts, pred_dicts_enc=None, pred_dicts_aux=None, dense_heatmap_aux=None, pred_map=None):

        loss_heatmap = torch.tensor(0.0).cuda()
        if self.model_cfg.TRAIN_BOX:
            device = dense_heatmap.device
            heatmap_list = []
            for bs in range(len(gt_labels_det)):
        
                gt_bboxes_3d = gt_boxes_det[bs]
                gt_labels_3d = gt_labels_det[bs] -1 

                self.feature_map_stride = 1
                feature_map_size = (180,180) #(self.grid_size[:2] // self.feature_map_stride)
                if self.one_channel:
                    heatmap = gt_bboxes_3d.new_zeros(1, feature_map_size[1], feature_map_size[0])
                else:
                    heatmap = gt_bboxes_3d.new_zeros(10, feature_map_size[1], feature_map_size[0])
                # heatmap = gt_bboxes_3d.new_zeros(1, feature_map_size[1], feature_map_size[0])
                for idx in range(len(gt_bboxes_3d)):
                    width = gt_bboxes_3d[idx][3]
                    length = gt_bboxes_3d[idx][4]
                    width = width / self.voxel_size[0] 
                    length = length / self.voxel_size[1] 
                    if width > 0 and length > 0:
                        radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), 0.1)[0]
                        radius = max(2, int(radius))
                        x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                        coor_x = (x - self.grid_conf['xbound'][0]) / self.voxel_size[0] 
                        coor_y = (y - self.grid_conf['ybound'][0]) / self.voxel_size[1]  

                        center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                        center_int = center.to(torch.int32)
                        if self.one_channel:
                            centernet_utils.draw_gaussian_to_heatmap(heatmap[0], center_int, radius)
                        else:
                            centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)
                heatmap_list.append(heatmap)
            heatmap = torch.stack(heatmap_list)

            # if self.model_cfg.get('DEEP_ENC_LOSS',False):
            #     loss_heatmap = self.loss_heatmap(
            #         clip_sigmoid(dense_heatmap),
            #         heatmap.repeat(dense_heatmap.shape[0],1,1,1),
            #     ).sum() / max(heatmap.eq(1).float().sum().item(), 1)

            # elif self.model_cfg.get('LEARN_DET_QUERY',False):
            #     dense_heatmap = outputs['pred_mask']
            #     loss_heatmap = self.loss_heatmap(
            #         clip_sigmoid(dense_heatmap),
            #         heatmap.repeat(dense_heatmap.shape[0],1,1,1),
            #     ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
            # else:

            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(dense_heatmap),
                heatmap,
            ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
            # if 'bev_heat_pred' in batch_dict:
            #     # import pdb;pdb.set_trace()
            #     self.loss_heatmap_aux = self.loss_heatmap(
            #     clip_sigmoid(batch_dict['bev_heat_pred']),
            #     heatmap).sum() / max(heatmap.eq(1).float().sum().item(), 1)
            #     loss_heatmap += self.loss_heatmap_aux
    
            # if dense_heatmap_aux is not None:
            #     loss_heatmap_aux = self.loss_heatmap(
            #     clip_sigmoid(dense_heatmap_aux),heatmap,).sum() / max(heatmap.eq(1).float().sum().item(), 1)
            #     loss_heatmap += loss_heatmap_aux

            # if self.spatial_mask is not None:
            #     loss_heatmap_aux = self.loss_heatmap(
            #     clip_sigmoid(self.spatial_mask),heatmap.sum(1,True),).sum() / max(heatmap.eq(1).float().sum().item(), 1)
            #     loss_heatmap += loss_heatmap_aux
            #     self.loss_heatmap_aux = loss_heatmap_aux
            #     # self.loss_heatmap_aux = torch.tensor(0.0)
            # else:
            #     pass
            #     #self.loss_heatmap_aux = torch.tensor(0.0)

        assigned_gt = [[],[]]#self.criterion.indices
        loss_ce_box = torch.tensor(0.0).cuda()
        loss_det = torch.tensor(0.0).cuda()
        loss_det_aux = torch.tensor(0.0).cuda()
        loss_reg = torch.tensor(0.0).cuda()
        loss_reg_enc = torch.tensor(0.0).cuda()
        loss_vels = torch.tensor(0.0).cuda()
        # loss_ce = torch.tensor(0.0).cuda()
        # loss_dice = torch.tensor(0.0).cuda()
        # loss_mask_cls = torch.tensor(0.0).cuda()
        # loss_mask = torch.tensor(0.0).cuda()
        # loss_road = torch.tensor(0.0).cuda()
        # loss_line_cls = torch.tensor(0.0).cuda()
        loss_map = torch.tensor(0.0).cuda()
        # loss_mask = torch.tensor(0.0).cuda()
        # loss_line_aux = torch.tensor(0.0).cuda()
        num_count = 0
        indices_list = []

        # if not self.only_train_map:

        if self.model_cfg.get('TRAIN_BOX',True):
            if pred_dicts_enc is not None:
                loss_i, losses,indices = self.get_det_loss(gt_boxes_det,gt_labels_det,pred_dicts_enc,assigned_gt,is_enc=True)
                loss_ce_box += losses['loss_cls']
                loss_reg_enc += losses['loss_bbox']
                indices_list.append(indices)
                loss_vels += losses['loss_vels']
                loss_det += loss_i
                num_count += 1


            # import pdb;pdb.set_trace()
            for i in range(self.model_cfg.MASK_FORMER.DEC_LAYERS-1):
                pred_dict = {}
                if self.model_cfg.get('USE_LOCAL_DEC',False):
                    for key in ['outputs_coords', 'heatmap']:
                        pred_dict[key] = pred_dicts[key][:,i]
                else:
                    for key in ['center', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                        pred_dict[key] = pred_dicts[key][:,i]
                loss_i, losses, indices = self.get_det_loss(gt_boxes_det,gt_labels_det,pred_dict,assigned_gt)
                loss_ce_box += losses['loss_cls']
                loss_reg += losses['loss_bbox']
                loss_vels += losses['loss_vels']
                loss_det += loss_i
                num_count += 1
                indices_list.append(indices)

            if self.model_cfg.get('USE_AUG_LOSS'):
                for i in range(self.model_cfg.MASK_FORMER.DEC_LAYERS-1):
                    pred_dict = {}
                    if self.model_cfg.get('USE_LOCAL_DEC',False):
                        for key in ['outputs_coords', 'heatmap']:
                            pred_dict[key] = pred_dicts_aux[key][:,i]
                    else:
                        for key in ['center', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                            pred_dict[key] = pred_dicts_aux[key][:,i]
                    loss_aux_i, losses, _ = self.get_det_loss(gt_boxes_det,gt_labels_det,pred_dict,assigned_gt,is_aug=True)
                    loss_det += loss_aux_i


        loss_line_enc = torch.tensor(0.0)
        self.loss_seg_aux = torch.tensor(0.0)
        if self.model_cfg.TRAIN_MAP:
            gt_bev_masks = batch_dict["gt_masks_bev"][0]
            # if self.model_cfg.get('USE_BEVSEG_HEAD',False):
            #     loss_line = batch_dict['map_loss']

            # elif self.model_cfg.get('ONLY_TRAIN_MAP',False) and not self.model_cfg.get('USE_TOP_SEG',False):
            #     label_weight = torch.ones_like(gt_bev_masks)

            #     for idx in range(self.model_cfg.MASK_FORMER.DEC_LAYERS-1):
            #         point_logits = outputs['pred_lines'][idx].view(batch_dict['batch_size'],6, 200,200)

            #         if self.model_cfg.get('USE_DSVT_MAP_LOSS',False):
            #             loss_line_i = self.map_cls(point_logits.flatten(2),gt_bev_masks.flatten(2),label_weight.flatten(2)).mean(-1)
            #             loss_line += loss_line_i.sum()
            #         if self.enc_pred_line is not None:
            #             loss_line_i = self.map_cls(self.enc_pred_line.flatten(2),gt_bev_masks.repeat(self.enc_pred_line.shape[0],1,1,1).flatten(2),label_weight.flatten(2)).mean(-1)
            #             loss_line += loss_line_i.sum()

            # else:
            if self.model_cfg.get('MAP_IGNORE_HEAT',False):
                label_weight = (heatmap.sum(1) <= 0.1)[:,None].repeat(1,6,1,1).float()
                label_weight = self.maskslicer(label_weight).transpose(2,3)
            else:
                label_weight = torch.ones_like(gt_bev_masks)
            
            # if self.enc_pred_line is not None:
            #     loss_line_i = self.map_cls(self.enc_pred_line.flatten(2),gt_bev_masks.repeat(self.enc_pred_line.shape[0],1,1,1).flatten(2),label_weight.flatten(2)).mean(-1)
            #     loss_line += loss_line_i.sum()
            if 'pred_lines' in outputs.keys():
                for idx in range(self.model_cfg.MASK_FORMER.DEC_LAYERS-1):
                    point_logits = outputs['pred_lines'][idx].view(batch_dict['batch_size'],6, 200,200)

                    # if self.model_cfg.get('USE_DSVT_MAP_LOSS',False):
                    loss_map_i = self.map_cls(point_logits.flatten(2),gt_bev_masks.flatten(2),label_weight.flatten(2)).mean(-1)
                    loss_map += loss_map_i.sum()

                    # if self.model_cfg.get('USE_AUG_LOSS',False) and idx <= (self.model_cfg.MASK_FORMER.DEC_LAYERS-2):
                    #     # import pdb;pdb.set_trace() 
                    #     point_logits = outputs['pred_lines_aux'][idx].view(batch_dict['batch_size'],6, 200,200)
                    #     # import pdb;pdb.set_trace() 
                    #     # if self.model_cfg.get('USE_DSVT_MAP_LOSS',False):
                    #         # for i in range(6):
                    #     label_weight = torch.ones_like(gt_bev_masks)
                    #     loss_line_i = self.map_cls(point_logits.flatten(2),gt_bev_masks.flatten(2),label_weight.flatten(2)).mean(-1)
                    #     loss_line_aux += loss_line_i.sum()

                    # if 'bev_seg_pred' in batch_dict:
                    #     point_logits = batch_dict['bev_seg_pred']
                    #     self.loss_seg_aux = self.map_cls(point_logits.flatten(2),gt_bev_masks.flatten(2),label_weight.flatten(2)).mean(-1).sum()
                    #     #loss_line += self.loss_seg_aux

                    # if 'bev_mask_pred' in batch_dict:
                    #     point_logits = batch_dict['bev_mask_pred']
                    #     gt_det_mask_full = torch.zeros_like(point_logits)
                    #     gt_det_mask_full[:,:batch_dict['gt_det_mask'].shape[1]] = batch_dict['gt_det_mask']
                    #     mask_weight = torch.ones_like(gt_det_mask_full)
                    #     self.loss_seg_aux = self.map_cls(point_logits.flatten(2),gt_det_mask_full.flatten(2),mask_weight.flatten(2)).mean(-1).sum()
                    #     loss_line += self.loss_seg_aux

                    # else:
                    #     src_logits = outputs["pred_logits_line"][idx].float()
                    #     cls_labels = [tgt['map_labels'].long() for tgt in targets]
                    #     target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
                    #     for bs in range(target_classes.shape[0]):
                    #         target_classes[bs][cls_labels[bs].long()] = cls_labels[bs] +1

                    #     loss_line_cls_i = sigmoid_focal_loss(src_logits, F.one_hot(target_classes,7).float()[...,1:]).mean()
                    #     loss_line_cls += loss_line_cls_i
                    #     point_logits_list = []
                    #     for bs in range(point_logits.shape[0]):
                    #         point_logits_list.append(point_logits[bs][cls_labels[bs]])
                    #     point_logits = torch.cat(point_logits_list)
                    #     # import pdb;pdb.set_trace()
                    #     # self.plt_show([point_logits[2].reshape(200,200).detach(),point_labels[2].reshape(200,200).detach()])
                    #     point_labels = torch.stack([tgt['map_masks'] for tgt in targets]).view(-1,40000)
                    #     mask_loss, _ = sigmoid_ce_loss_v2(point_logits, point_labels, num_masks=num_masks, weight=None,with_road=False)
                    #     loss_dice = dice_loss(point_logits, point_labels, src_logits.shape[0]*6)
                    #     loss_line_i = mask_loss + loss_dice
                    #     loss_line += loss_line_i

            if self.model_cfg.get('USE_TOP_SEG',False): 
                loss_map_enc = self.map_cls(pred_map.flatten(2),gt_bev_masks.flatten(2),label_weight.flatten(2)).mean(-1).sum()
            else:
                loss_map_enc = 0


        # mask_loss = loss_mask + loss_mask_cls
        # line_loss = loss_map + loss_map_cls + loss_line_aux

        loss = loss_map*self.map_weight + loss_det * self.det_weight + loss_heatmap*self.heatmap_weight + loss_mask*self.heatmap_weight + \
               loss_map_enc * self.map_enc_weight
        # loss_ce = loss_mask_cls*0 + loss_line_cls
        return loss, loss_heatmap, loss_map, loss_ce_box,loss_reg,loss_reg_enc,loss_det,loss_det_aux,loss_map_enc
    
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts,assigned_gt,is_enc):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict, assigned_gt,is_enc)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        indices = res_tuple[7]
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap, indices

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict,assigned_gt,is_enc):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]["pred_boxes"]

        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)
        gt_labels_3d = gt_labels_3d -1

        assigned_gt_inds, ious_all, matched_row_inds, matched_col_inds = self.bbox_assigner.assign(
        bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
        score, self.point_cloud_range,)


        if gt_bboxes_tensor.shape[0] > 0 and len(matched_col_inds) > 0:
            ious = torch.zeros(bboxes_tensor.shape[0]).cuda()
            ious[matched_row_inds] = ious_all[matched_row_inds, matched_col_inds]
        else:
            ious = torch.zeros(bboxes_tensor.shape[0]).cuda()

        indices = [matched_row_inds, matched_col_inds]
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        if is_enc:
            bbox_targets = torch.zeros([num_proposals, self.code_size]).to(score.device)
            bbox_weights = torch.zeros([num_proposals, self.code_size]).to(score.device)
        else:
            if self.model_cfg.get('USE_LOCAL_DEC',False):
                bbox_targets = torch.zeros([num_proposals, 9]).to(score.device)
                bbox_weights = torch.zeros([num_proposals, 9]).to(score.device)
            else:
                bbox_targets = torch.zeros([num_proposals, self.code_size]).to(score.device)
                bbox_weights = torch.zeros([num_proposals, self.code_size]).to(score.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += 10 #self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            # if self.model_cfg.get('USE_LOCAL_DEC',False) and not is_enc:
            #     pos_bbox_targets = self.sigmoid_encode_bbox(pos_gt_bboxes)
            # else:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        heatmap = torch.zeros(128,128)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None],indices)

    def get_det_loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts,assigned_gt, is_enc=False, is_aug=False,**kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap,indices = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts,assigned_gt,is_enc)

        loss_dict = dict()
        loss_all = 0
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, 10)

        one_hot_targets = torch.zeros(*list(labels.shape), 10+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)
        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        if is_enc:
            reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)
        else:
            if self.model_cfg.get('USE_LOCAL_DEC',False):
                reg_weights = bbox_weights * bbox_weights.new_tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,0.2])
            else:
                reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets)
        loss_dict["loss_vels"] = (loss_bbox[...,-2:] * reg_weights[...,-2:]).sum() / max(num_pos, 1) * self.loss_bbox_weight
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox * self.loss_bbox_weight
        if self.model_cfg.get('DECODE_NOT_REG', False):
            loss_all = loss_all + loss_cls * self.loss_cls_weight 
        else:
            if is_aug:
                loss_all = loss_all + loss_bbox * self.loss_bbox_weight
            else:
                loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict, indices
    
    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:9]
        return targets
  
    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False, filter_score=None):
        # import pdb;pdb.set_trace()
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = filter_score if filter_score is not None else post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[0]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts,filter=True,is_enc=False):
        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid().detach().clone()
        if self.one_channel:
            self.query_labels = batch_score[0].max(0)[1][None]
        one_hot = F.one_hot(
            self.query_labels, num_classes=10
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"].detach().clone()
        batch_height = preds_dicts["height"].detach().clone()
        batch_dim = preds_dicts["dim"].detach().clone()
        batch_rot = preds_dicts["rot"].detach().clone()
        batch_vel = None
        batch_iou = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=filter,
        )

        post_process_cfg = self.model_cfg.POST_PROCESSING
        
        if self.model_cfg.POST_PROCESSING.NMS_CONFIG.ENABLED:
            """
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores= ret_dict[0]['pred_scores'], box_preds = ret_dict[0]['pred_boxes'],
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH)

            for keys in ret_dict[0].keys():
                ret_dict[0][keys] = ret_dict[0][keys][selected]

            for k in range(batch_size):
                ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1
            """

            # if self.dataset_name == "nuscenes":
            self.tasks = [
                dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                dict(num_class=1, class_names=["pedestrian"], indices=[8], radius=0.175),
                dict(num_class=1,class_names=["traffic_cone"],indices=[9],radius=0.175),
            ]

            new_ret_dict = []
            for i in range(batch_size):
                boxes3d = ret_dict[i]["pred_boxes"]
                scores = ret_dict[i]["pred_scores"]
                labels = ret_dict[i]["pred_labels"]
                cmask = ret_dict[i]['cmask']
                # IOU refine 
                if self.model_cfg.POST_PROCESSING.get('USE_IOU_TO_RECTIFY_SCORE', False) and batch_iou is not None:
                    pred_iou = torch.clamp(batch_iou[i][0][cmask], min=0, max=1.0)
                    IOU_RECTIFIER = scores.new_tensor(self.model_cfg.POST_PROCESSING.IOU_RECTIFIER)
                    if len(IOU_RECTIFIER) == 1:
                        IOU_RECTIFIER = IOU_RECTIFIER.repeat(self.num_classes)
                    scores = torch.pow(scores, 1 - IOU_RECTIFIER[labels]) * torch.pow(pred_iou, IOU_RECTIFIER[labels])
                
                keep_mask = torch.zeros_like(scores)
                for task in self.tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    if task["radius"] > 0:
                        top_scores = scores[task_mask]
                        boxes_for_nms = boxes3d[task_mask][:, :7].clone().detach()
                        task_nms_config = copy.deepcopy(self.model_cfg.POST_PROCESSING.NMS_CONFIG)
                        task_nms_config.NMS_THRESH = task["radius"]
                        task_keep_indices, _ = model_nms_utils.class_agnostic_nms(
                                box_scores=top_scores, box_preds=boxes_for_nms,
                                nms_config=task_nms_config, score_thresh=post_process_cfg.SCORE_THRESH)
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())
                    
                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                        keep_mask[keep_indices] = 1
                keep_mask = keep_mask.bool()
                ret = dict(pred_boxes=boxes3d[keep_mask], pred_scores=scores[keep_mask], pred_labels=labels[keep_mask])
                new_ret_dict.append(ret)

            for k in range(batch_size):
                new_ret_dict[k]['pred_labels'] = new_ret_dict[k]['pred_labels'].int() + 1

            return new_ret_dict

        else:
            for k in range(batch_size):
                ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

            return ret_dict

    def reorder_gt_boxes_det(self, batch_dict, gt_boxes,gt_labels,only_vehicle=False):

        # All classes [ 'car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        # vehicle_classes = ['car', 'truck','construction_vehicle', 'bus', 'trailer',             'motorcycle','bicycle']
        gt_boxes_list = []
        gt_labels_list = []
        batch_size = batch_dict['batch_size']
        # import pdb;pdb.set_trace()
        for bs_idx in range(batch_size):
            if 1 in batch_dict['gt_boxes_index'][0].unique():
                mask = batch_dict['gt_boxes_index'][0] == (batch_size - 1 - bs_idx) # convert to ...,-3,-2,-1,0
                label_mask = gt_labels[0] <= 10
                mask = label_mask
            else:
                label_mask = gt_labels[0] <= 10
                mask = label_mask
            gt_boxes_list.append(gt_boxes[:,mask][0])
            gt_labels_list.append(gt_labels[:,mask][0].long())
        num_max_rois = max([len(boxes) for boxes in gt_boxes_list])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = gt_boxes_list[0]

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        for bs_idx in range(batch_size):
            num_boxes = len(gt_boxes_list[bs_idx])

            rois[bs_idx, :num_boxes, :] = gt_boxes_list[bs_idx]

        return rois, gt_boxes_list, gt_labels_list

    def loss_voxel(self, output_voxels, target_voxels, tag=0):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels.contiguous(), ignore=255)

        return loss_dict

    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):

        def fast_hist(pred, label, max_label=18):
            pred = copy.deepcopy(pred.flatten())
            label = copy.deepcopy(label.flatten())
            bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
            return bin_count[:max_label ** 2].reshape(max_label, max_label)

        _, H, W, D = gt.shape
        # pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int32)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ

    def forward(self, batch_dict, mask=None):
        x = batch_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.voxel_indices, self.spatial_indices,self.num_voxels = voxel_indices, spatial_indices, num_voxels

        feats = batch_dict['spatial_features_2d']

        bev_pos,bev_embed,bev_map_embed, bev_embed_dec = self.get_bev_pos_embed(feats)
        if self.query_channel == self.hidden_channel:
            bev_embed_dec = bev_embed


        query_feat, heatmap, query_embed, query_labels,query_pos, bev_pos, dense_heatmap,top_proposals_index,query_score, pred_map, _,enc_pred_line, feats_cls, top_seg_index, bg_index  = \
            self.get_query_embed_enc_only(batch_dict, feats, bev_pos, bev_embed_dec)
        num_det_seg = query_feat.shape[-1]
        if self.model_cfg.get('USE_TOP_SEG',False):
            top_index  = torch.cat([top_proposals_index.reshape(-1),top_seg_index.reshape(-1)])
        else:
            top_index  = top_proposals_index.reshape(-1)
        if self.model_cfg.get('TRAIN_OCC',False):
            self.bev_h = 180
            self.bev_w = 180
            if self.model_cfg.get('USE_11_HIGHT',False):
                voxel_4x_index = batch_dict['multi_scale_3d_features']['x_conv3'].indices[:,[3,2,1]].long() # xyz
                empty_voxel = torch.zeros([360,360,11]).cuda()
                empty_voxel[voxel_4x_index[:,0],voxel_4x_index[:,1],voxel_4x_index[:,2]]=1
                voxel_8x = F.max_pool3d(empty_voxel.permute(2,0,1)[None], kernel_size=(1,2,2), stride=(1,2,2),padding=(0,0,0))[0].permute(1,2,0)
                index_xyz = voxel_8x.nonzero()
                self.bev_z = 11
            else:
                index_zxy = batch_dict['multi_scale_3d_features']['x_conv4'].indices[:,1:]
                index_xyz = index_zxy[:,[2,1,0]].long() # xyz
                self.bev_z = 5
                cur_coords = batch_dict['multi_scale_3d_features']['x_conv4'].indices

                voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=1,
                    voxel_size=[0.6,0.6,1.6],
                    point_cloud_range=self.point_cloud_range
                )

            occ_pos, occ_pos_3d, occ_embed = self.get_occ_pos_embed(feats, self.bev_z)

            query_feat_occ, occ_index  = self.get_occ_query_feat(feats,index_xyz)
            query_embed_occ = occ_embed[:,:,index_xyz[:,0],index_xyz[:,1],index_xyz[:,2]]
            query_pos_occ = voxel_xyz
            # pcd_np_cor = query_pos * self.voxel_size[None, :] + self.point_cloud_range[:3][None, :]
            inside_scene_mask_x = torch.logical_and(query_pos_occ[:, 0] >= -51.2, query_pos_occ[:, 0] <= 51.2)
            inside_scene_mask_y = torch.logical_and(query_pos_occ[:, 1] >= -51.2, query_pos_occ[:, 1] <= 51.2)
            inside_scene_mask_z = torch.logical_and(query_pos_occ[:, 2] >= -5.0, query_pos_occ[:, 2] <= 3.0)
            inside_scene_mask = np.logical_and.reduce((inside_scene_mask_x.cpu().numpy(), inside_scene_mask_y.cpu().numpy(), inside_scene_mask_z.cpu().numpy()))
            inside_scene_mask = torch.from_numpy(inside_scene_mask).cuda()
            index_xyz = index_xyz[inside_scene_mask]
            query_feat_occ  = query_feat_occ[...,inside_scene_mask]
            query_embed_occ = query_embed_occ[...,inside_scene_mask]
            query_pos_occ = query_pos_occ[inside_scene_mask][None]
            occ_index = occ_index[inside_scene_mask].reshape(-1)
            if not self.model_cfg.get('OCC_USE_BEV_FEAT',True):
                query_feat_occ  = self.mask_embed.weight.view(1, 1, self.query_channel).expand(1, query_feat_occ.shape[-1], self.query_channel).transpose(1,2)

            dense_heatmap_aux = None
 
            query_feat = torch.cat([query_feat,query_feat_occ],-1)
            query_embed = torch.cat([query_embed,query_embed_occ],-1)
            query_pos = torch.cat([query_pos,query_pos_occ[...,:2]],1)
            top_index = torch.cat([top_index,occ_index])
            # query_feat,map_features = self.pixel_decoder.forward_features(feats,bev_embed,query_feat,query_embed,occ_pos[...,:2])

        self.enc_pred_line = enc_pred_line
        if self.model_cfg.get('USE_VM_ENC',False):
            query_feat,map_features  = self.pixel_decoder(feats,top_index,bg_index,query_feat.unsqueeze(1).transpose(2,3))
        else:
            query_feat,map_features, memory = self.pixel_decoder.forward_features(feats,bev_embed,query_feat,query_embed,query_pos,top_index)

        loss_occ_sum = 0
        loss_occ_sum_aux = torch.tensor(0.0).cuda()

        if self.model_cfg.get('TRAIN_OCC',False):
    
            vox_feats = torch.empty((1, self.bev_h, self.bev_w, self.bev_z, self.query_channel), device=query_feat.device)
            vox_mask = torch.ones((1, self.bev_h, self.bev_w, self.bev_z), device=query_feat.device).bool()
            vox_mask[:,index_xyz[:,0],index_xyz[:,1],index_xyz[:,2]] = False
            query_feat_occ = query_feat[:,num_det_seg:]
            vox_feats[:,index_xyz[:,0],index_xyz[:,1],index_xyz[:,2], :] = query_feat_occ
            vox_feats[vox_mask,:] = self.mask_embed.weight.view(1, 1, self.query_channel).expand(1, vox_mask.sum(), self.query_channel)
            vox_feats[vox_mask,:] = vox_feats[vox_mask,:] + occ_embed.permute(0,2,3,4,1)[vox_mask]
            vox_feats_flatten = vox_feats.reshape(1, -1, self.query_channel)

            query_feat_occ = self.occ_attn[0](vox_feats_flatten,
                        query_pos=occ_embed.permute(0,2,3,4,1).reshape(1,-1,self.query_channel),reference_points=occ_pos)
            query_feat_occ = self.occ_ffn[0](query_feat_occ)

            query_feat_occ = self.occ_attn[1](query_feat_occ,
                        query_pos=occ_embed.permute(0,2,3,4,1).reshape(1,-1,self.query_channel),reference_points=occ_pos)
            query_feat_occ = self.occ_ffn[1](query_feat_occ)
            query_feat_occ = self.cubeslicer(query_feat_occ.reshape(1,180,180,5*256).permute(0,3,1,2)).permute(0,3,2,1).reshape(1,171,171,self.bev_z,self.query_channel)
            pred_occ = self.voxel_mlp_head(query_feat_occ).permute(0,4,1,2,3).contiguous()
            gt_occ = batch_dict['gt_occ']
            pred_occ = F.interpolate(pred_occ, size=[512, 512, 40], mode='trilinear', align_corners=False).contiguous()

            if self.training:
                loss_occ = self.loss_voxel(pred_occ,gt_occ)
                for key in loss_occ.keys():
                    loss_occ_sum += loss_occ[key]

        query_feat = query_feat[:,:num_det_seg]
        query_embed = query_embed[:,:num_det_seg]
        query_pos = query_pos[:,:num_det_seg]
        map_feat  = query_feat[:,200:num_det_seg]
        map_feat = self.map_norm_dec(map_feat)
        map_feat = self.map_embed_dec(map_feat)
        num_seg_query = self.model_cfg.SEG_QUERY_NUM
        b,_,h,w = map_features.shape
        if num_seg_query==6:
            output_map = torch.einsum("bqc,bchw->bqhw", map_feat.reshape(-1,1,map_feat.shape[-1]), map_features).permute(1,0,2,3).contiguous()
        else:
            split = num_seg_query//6
            map_feats = map_features#.reshape(b,c,split,h//split,w)
            line = map_feat.reshape(split,6,map_feat.shape[-1])
            outputs_line_list = []
            for i in range(split):
                outputs_line = torch.einsum("bqc,bchw->bqhw", line[i:i+1], map_feats)
                outputs_line = outputs_line.reshape(b,6,split,h//split,w)
                outputs_line_list.append(outputs_line[:,:,i])
            output_map = torch.cat(outputs_line_list,2)

        _, gt_boxes_list_det, gt_labels_det = self.reorder_gt_boxes_det(batch_dict,batch_dict['gt_boxes'].clone(),batch_dict['gt_boxes'][...,-1].clone())
        predictions = {}
        predictions['query_feat'] = [query_feat[:,:200].transpose(1,2)]
        predictions['pred_boxes'] = None
        predictions['pred_lines'] = [output_map]
        pred_dict_enc = None
        dense_heatmap_aux = None
        pred_dicts, pred_dicts_aux = self.get_pred_dict_dec(predictions,query_pos,heatmap,top_proposals_index,dense_heatmap[-1:],predictions['pred_boxes'])

        if self.training:


            loss,loss_heatmap, loss_line,loss_ce_box,loss_reg,loss_reg_enc,loss_det, loss_det_aux,loss_seg_enc = \
                self.get_loss_enc_only(
                batch_dict,predictions,gt_labels_det,gt_boxes_list_det,dense_heatmap,pred_dicts, pred_dict_enc,pred_dicts_aux,dense_heatmap_aux,pred_map)

            if 'loss_bbox_aug' in batch_dict:
                loss_det_aux = batch_dict['loss_bbox_aug']
                loss = loss + batch_dict['loss_bbox_aug']

            if self.loss_seg_aux != 0:
                loss = loss + self.model_cfg.get('BEVFUSER_SEG_LOSS_WEIGHT', 1) * self.loss_seg_aux

            batch_dict['loss'] = loss + loss_occ_sum*self.model_cfg.get('OCC_LOSS_WEIGHT',1.0)
            tb_dict = {}
            # tb_dict['loss_cls_mask'] = loss_mask_cls
            # tb_dict['loss_cls_line'] = loss_line_cls
            tb_dict['loss_heatmap'] = loss_heatmap
            # import pdb;pdb.set_trace()
            try:
                tb_dict['loss_heatmap_aux'] = self.loss_heatmap_aux
            except:
                tb_dict['loss_heatmap_aux'] = torch.tensor(0.0)
            # tb_dict['loss_mask'] = loss_mask
            tb_dict['loss_occ'] = torch.tensor(loss_occ_sum)
            # tb_dict['loss_road'] = loss_road
            tb_dict['loss_line'] = loss_line
            if self.only_train_occ:
                tb_dict['loss_cls'] = loss_occ['loss_voxel_ce_0']
            else:
                tb_dict['loss_cls'] = loss_ce_box
            tb_dict['loss_reg'] = loss_reg
            tb_dict['loss_reg_enc'] = loss_reg_enc
            tb_dict['loss_vels'] = loss_det
            tb_dict['loss_det_aux'] = loss_det_aux
            tb_dict['loss_occ_sum_aux'] = loss_occ_sum_aux
            if self.model_cfg.get('USE_TOP_SEG',False):
                tb_dict['loss_seg_aux'] = loss_seg_enc
            # else:
            #     tb_dict['loss_seg_aux'] = loss_seg_aux
            tb_dict['loss_bev_seg_aux'] = self.loss_seg_aux
            batch_dict['tb_dict'] = tb_dict

        else:
            bboxes = self.get_bboxes(pred_dicts)
            batch_dict['final_box_dicts'] = bboxes
            batch_dict['map_results'] = [{
                            'masks_bev': output_map.sigmoid(),
                            'gt_masks_bev': batch_dict['gt_masks_bev'][0,0]
                            }]

            if self.model_cfg.get('TRAIN_OCC',False):

                SC_metric, _ = self.evaluation_semantic(pred_occ, gt_occ, eval_type='SC', visible_mask=None)
                SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_occ, gt_occ, eval_type='SSC', visible_mask=None)
                batch_dict['occ_results'] = {}
                batch_dict['occ_results'].update({
                                    'SC_metric': SC_metric,
                                    'SSC_metric': SSC_metric
                                            })

            return batch_dict




