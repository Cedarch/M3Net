from typing import Optional
import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
import copy
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
from pcdet.ops.multi_scale_deformable_attn.multi_scale_deform_attn_utils import MSDeformAttn
from fvcore.nn import flop_count, parameter_count
import fvcore.nn.weight_init as weight_init

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def calculate_birds_eye_view_parameters(
    x_bounds, y_bounds, z_bounds
):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position: Bird's-eye view start position for area of interest.
        bev_dimension: Bird's-eye view dimension for area of interest.
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long,
    )

    return bev_resolution, bev_start_position, bev_dimension

class SElayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SElayer,self).__init__()
        self.conv = nn.Conv2d(channel, channel, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x, y=None):
        b, c, _, _ = x.size()
        x = self.conv(x)
        if y is None:
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:
            return  x * y

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class BevFeatureSlicer(nn.Module):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False
            bev_resolution, self.bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
            )

            self.map_bev_resolution, self.map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                self.map_bev_start_position[0], map_grid_conf['xbound'][1], self.map_bev_resolution[0])

            self.map_y = torch.arange(
                self.map_bev_start_position[1], map_grid_conf['ybound'][1], self.map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- self.bev_start_position[0])
            self.norm_map_y = self.map_y / (- self.bev_start_position[1])

            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_x, self.norm_map_y, indexing='ij'), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class TCSLayer(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.ln = nn.LayerNorm(dim)

        self.gate1 = nn.Linear(dim, dim)
        self.gate2 = nn.Linear(dim, dim)
        if self.config.get('SPLIT_GATE',False):
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            if self.config.get('TRAIN_OCC',False):
                self.gate3 = nn.Linear(dim, dim)
                self.proj3 = nn.Linear(dim, dim)
        else:
            if self.config.get('TRAIN_OCC',False):
                self.gate3 = nn.Linear(dim, dim)
                self.proj = nn.Linear(dim, 3*dim)
            else:
                self.proj = nn.Linear(dim, 2*dim)


    def forward(self, x):
        if self.config.get('SPLIT_GATE',False):
            gate_det = F.gelu(self.proj1(x))
            gate_seg = F.gelu(self.proj2(x))
            if self.config.get('TRAIN_OCC',False):
                gate_occ = F.gelu(self.proj3(x))
        else:
            if self.config.get('TRAIN_OCC',False):
                gate_det,gate_seg,gate_occ = F.gelu(self.proj(x)).chunk(3,-1)
            else:
                gate_det,gate_seg = F.gelu(self.proj(x)).chunk(2,-1)
        if self.config.get('USE_SIGMOID_GATE',False):
            gate_det = self.gate1(gate_det).sigmoid()
            gate_seg = self.gate2(gate_seg).sigmoid()
            if self.config.get('TRAIN_OCC',False):
                gate_occ = self.gate3(gate_occ).sigmoid()
            else:
                gate_occ = None
        else:
            gate_det = self.gate1(gate_det)
            gate_seg = self.gate2(gate_seg)
            if self.config.get('TRAIN_OCC',False):
                gate_occ = self.gate3(gate_occ)
            else:
                gate_occ = None
        return gate_det, gate_seg, gate_occ

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 memory_two_ffn=False, query_two_ffn=False,
                 channel_gating=False, spatial_gating=False,
                 use_spatial_mask=False,config=None):
        super().__init__()

        # self attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.query_attn = SelfAttentionLayer(
                                    d_model=d_model,
                                    nhead=n_heads,
                                    dropout=0.0,
                                    normalize_before=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_query = nn.Dropout(dropout)
        self.norm_query = nn.LayerNorm(d_model)

        self.linear1_2 = nn.Linear(d_model, d_ffn)
        self.activation_2 = _get_activation_fn(activation)
        self.dropout2_2 = nn.Dropout(dropout)
        self.linear2_2 = nn.Linear(d_ffn, d_model)
        self.dropout3_2 = nn.Dropout(dropout)
        self.norm2_2 = nn.LayerNorm(d_model)

        self.linear1_3 = nn.Linear(d_model, d_ffn)
        self.activation_3 = _get_activation_fn(activation)
        self.dropout2_3 = nn.Dropout(dropout)
        self.linear2_3 = nn.Linear(d_ffn, d_model)
        self.dropout3_3 = nn.Dropout(dropout)
        self.norm2_3 = nn.LayerNorm(d_model)

        self.linear1_4 = nn.Linear(d_model, d_ffn)
        self.activation_4 = _get_activation_fn(activation)
        self.dropout2_4 = nn.Dropout(dropout)
        self.linear2_4 = nn.Linear(d_ffn, d_model)
        self.dropout3_4 = nn.Dropout(dropout)
        self.norm2_4 = nn.LayerNorm(d_model)

        self.config = config

        self.gate = TCSLayer(d_model,config)



    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


    def forward_ffn_3(self, src):
        src2 = self.linear2_3(self.dropout2_3(self.activation_3(self.linear1_3(src))))
        src = src + self.dropout3_3(src2)
        src = self.norm2_3(src)
        return src


    def forward(self, query, query_pos, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, 
                spatial_mask=None,query_index=None,layer_index=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points[:,:32400], src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        gate, gate_det,gate_seg = None, None, None
        # ffn
        num_seg_query = self.config.get('SEG_QUERY_NUM')+200
        src1 = self.forward_ffn(src)

        if self.config.get('USE_TCS',False):
            src = src1
            if self.config.get('TCS_WITH_CA',False):

                    gate_det,gate_seg,gate_occ = self.gate(src)
                    src_det,src_seg = src*gate_det, src*gate_seg
                    query2_det = self.cross_attn(self.with_pos_embed(query[:,:200], query_pos[:,:200]), 
                                            reference_points[:,32400:32600], src_det, 
                                            spatial_shapes, level_start_index, padding_mask)
                    query2_seg = self.cross_attn(self.with_pos_embed(query[:,200:num_seg_query], query_pos[:,200:num_seg_query]), 
                                            reference_points[:,32600:32400+num_seg_query], src_seg, 
                                            spatial_shapes, level_start_index, padding_mask)

                    if self.config.get('TRAIN_OCC',False):
                        src_occ = src*gate_occ
                        query2_occ = self.cross_attn(self.with_pos_embed(query[:,num_seg_query:], query_pos[:,num_seg_query:]), 
                                                reference_points[:,32400+num_seg_query:], src_occ, 
                                                spatial_shapes, level_start_index, padding_mask)
                        query2 = torch.cat([query2_det,query2_seg,query2_occ],1)
                    else:
                        query2 = torch.cat([query2_det,query2_seg],1)
            else:

                src_index = src[:,query_index]
                gate_det,gate_seg,gate_occ = self.gate(src_index)
                query_det = src[:,query_index[:200]]
                query_seg = src[:,query_index[200:num_seg_query]]
                if self.config.get('TRAIN_OCC',False):
                    query_occ = src[:,query_index[num_seg_query:]]
                    query_occ = query_occ*gate_occ[:,num_seg_query:]
                    query_det,query_seg = query_det*gate_det[:,:200], query_seg*gate_seg[:,200:num_seg_query]
                    query2 = torch.cat([query_det,query_seg,query_occ],1)
                else:
                    query_det,query_seg = query_det*gate_det[:,:200], query_seg*gate_seg[:,200:num_seg_query]
                    query2 = torch.cat([query_det,query_seg],1)

            query = query + self.dropout_query(query2)
            query = self.norm_query(query)
            query = self.query_attn(query.transpose(0,1), query_pos=query_pos.transpose(0,1)).transpose(0,1)
            query = self.forward_ffn_3(query)


        return query, src, gate, gate_det, gate_seg

class MSEncoderWithQuery(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_ * W_, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,spatial_mask=None,query_refpoint=None,
                query=None,query_embed=None,query_index=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        if query_refpoint is not None:
            reference_points = torch.cat([reference_points,query_refpoint.unsqueeze(2)/180],1)
        output_list = []
        for idx, layer in enumerate(self.layers):
            query, output, gate, gate_det,gate_seg = layer(query, query_embed, output, pos, 
                                    reference_points, spatial_shapes, level_start_index, padding_mask, spatial_mask, query_index,idx)
            output_list.append(output)
        return query, torch.cat(output_list), gate, gate_det,gate_seg

class MSDeformAttnTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4, 
                 memory_two_ffn=False, query_two_ffn=False, 
                 channel_gating=False, 
                 spatial_gating=False, use_spatial_mask=False,config=None
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points,
                                                            memory_two_ffn, query_two_ffn,
                                                            channel_gating,spatial_gating, 
                                                            use_spatial_mask, config)
        self.encoder = MSEncoderWithQuery(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, query_feat, query_embed, query_pos, query_index=None):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten.append(query_feat.transpose(1,2))
        query_embed = query_embed.transpose(1,2) + self.level_embed[lvl].view(1, 1, -1)
        # lvl_pos_embed_flatten.append(query_embed)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = None #torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        spatial_mask = None
        # (self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,spatial_mask=None,query_refpoint=None, query=None,query_embed=None):
        query, memory, gate, gate_det,gate_seg = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, spatial_mask, 
                              query_pos, query_feat.transpose(1,2), query_embed, query_index)
        return query, memory, spatial_shapes, level_start_index, gate, gate_det,gate_seg

class MSDeformAttnPixelDecoderEncOnly(nn.Module):
    def __init__(
        self,
        input_shape,
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_dim_feedforward=2048,
        transformer_enc_layers=6,
        conv_dim=256,
        mask_dim=256,
        transformer_in_features= ["res3", "res4", "res5"],
        common_stride=4,
        config=None
    ):
        super().__init__()
       
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features} 
            
        self.in_features = [k for k, v in input_shape.items()]        
        self.feature_channels = [v['channel'] for k, v in input_shape.items()] 
             
        self.transformer_in_features = [k for k, v in transformer_input_shape.items()]  
        transformer_in_channels = [v['channel'] for k, v in transformer_input_shape.items()] 
        self.transformer_feature_strides = [v['stride'] for k, v in transformer_input_shape.items()]  

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res3)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderWithQuery(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            enc_n_points=config.MASK_FORMER.get('NUM_REF_POINTS',4),
            config = config
        )
        N_steps = conv_dim // 2


        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.line_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.line_features)

        self.map_features = SElayer(conv_dim)
        self.heatmap_features = SElayer(conv_dim)

        weight_init.c2_xavier_fill(self.line_features)
     
        self.maskformer_num_feature_levels = 1  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]): # res2 -> fpn
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                                         nn.GroupNorm(32, conv_dim),
                                         nn.ReLU(inplace=True))

            output_conv = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, kernel_size=3,  stride=1,  padding=1),
                                        nn.GroupNorm(32, conv_dim),
                                        nn.ReLU(inplace=True))
            
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        if config.MASK_FORMER.get('FULL_MAP',False):
            map_grid_conf = {
                'ybound': [-50.0, 50.0, 0.25],
                'xbound': [-50.0, 50.0, 0.25],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }
        else:
            map_grid_conf = {
                'ybound': [-30.0, 30.0, 0.15],
                'xbound': [-15.0, 15.0, 0.15],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }

        motion_grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.grid_conf = {
            'xbound': [-54, 54, 0.6],
            'ybound': [-54, 54, 0.6],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }
        self.config = config
        # self.lineslicer = BevFeatureSlicer(grid_conf=self.grid_conf,map_grid_conf=map_grid_conf)
        self.maskslicer = BevFeatureSlicer(grid_conf=self.grid_conf,map_grid_conf=motion_grid_conf)
        self.res_bev_feat = config.get('RES_BEV_FEAT', False)


    def forward_features(self, features, srcs_emb, query_feat, query_embed, query_pos,query_index=None):
        srcs = []
        pos = []
        srcs.append(self.input_proj[0](features))
        pos.append(srcs_emb)
        
        query, y, spatial_shapes, level_start_index, gate, gate_det, gate_seg = self.transformer(srcs, pos, query_feat, query_embed, query_pos, query_index)
        bs = y.shape[0]
        self.transformer_num_feature_levels = 1
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        level_start_index = [0,180*180]
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        for i, z in enumerate([y[0]]):
            # z = self.up(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
            z = z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1])
            out.append(z)

        if self.res_bev_feat:
            if self.config.get('USE_TCS', False):
                map_features = self.map_features(out[-1][-1:] + srcs[0], gate_seg.transpose(2,1).reshape(1,out[-1][-1:].shape[1],180,180))
        else:
            map_features = self.map_features(out[-1][-1:] + srcs[0])
        map_features = self.maskslicer(map_features)


        return query, map_features, y[0][-1:].permute(0,2,1).reshape(features.shape[0],256,180,180)

    def forward(self, features, srcs_emb, query_feat, query_embed, query_pos,query_index=None):
        return self.forward_features(features, srcs_emb, query_feat, query_embed, query_pos,query_index)

    def flops(self, shape=(512, 180, 180)):

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, 
                                         inputs=(input,torch.zeros([1, 256, 180, 180]).cuda(), #srcs_emb
                                                 torch.zeros([1, 256, 230]).cuda(), #query_feat
                                                 torch.zeros([1, 256, 230]).cuda(), #query_embed,
                                                 torch.zeros([1, 230, 2]).cuda(), #query_pos,
                                                 torch.zeros([230]).cuda().long(), #query_index,
                                                 ))

        del model, input
        return f"params {params} GFLOPs {sum(Gflops.values())}"

class MSDeformAttnTransformerEncoderWithQuery(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4, 
                 memory_two_ffn=False, query_two_ffn=False, 
                 channel_gating=False, 
                 spatial_gating=False, use_spatial_mask=False,config=None
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayerWithQuery(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points,
                                                            memory_two_ffn, query_two_ffn,
                                                            channel_gating,spatial_gating, 
                                                            use_spatial_mask, config)
        self.encoder = MSEncoderWithQuery(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, query_feat, query_embed, query_pos, query_index=None):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten.append(query_feat.transpose(1,2))
        query_embed = query_embed.transpose(1,2) + self.level_embed[lvl].view(1, 1, -1)
        # lvl_pos_embed_flatten.append(query_embed)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = None #torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        spatial_mask = None
        # (self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,spatial_mask=None,query_refpoint=None, query=None,query_embed=None):
        query, memory, gate, gate_det,gate_seg = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, spatial_mask, 
                              query_pos, query_feat.transpose(1,2), query_embed, query_index)
        return query, memory, spatial_shapes, level_start_index, gate, gate_det,gate_seg
    
class MSDeformAttnTransformerEncoderLayerWithQuery(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 memory_two_ffn=False, query_two_ffn=False,
                 channel_gating=False, spatial_gating=False,
                 use_spatial_mask=False,config=None):
        super().__init__()

        # self attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.query_attn = SelfAttentionLayer(
                            d_model=d_model,
                            nhead=n_heads,
                            dropout=0.0,
                            normalize_before=False)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.memory_two_ffn = memory_two_ffn
        self.query_two_ffn = query_two_ffn
        self.channel_gating = channel_gating
        self.spatial_gating = spatial_gating
        self.use_spatial_mask = use_spatial_mask
        self.dropout_query = nn.Dropout(dropout)
        self.norm_query = nn.LayerNorm(d_model)

        self.linear1_2 = nn.Linear(d_model, d_ffn)
        self.activation_2 = _get_activation_fn(activation)
        self.dropout2_2 = nn.Dropout(dropout)
        self.linear2_2 = nn.Linear(d_ffn, d_model)
        self.dropout3_2 = nn.Dropout(dropout)
        self.norm2_2 = nn.LayerNorm(d_model)

        self.linear1_3 = nn.Linear(d_model, d_ffn)
        self.activation_3 = _get_activation_fn(activation)
        self.dropout2_3 = nn.Dropout(dropout)
        self.linear2_3 = nn.Linear(d_ffn, d_model)
        self.dropout3_3 = nn.Dropout(dropout)
        self.norm2_3 = nn.LayerNorm(d_model)

        self.linear1_4 = nn.Linear(d_model, d_ffn)
        self.activation_4 = _get_activation_fn(activation)
        self.dropout2_4 = nn.Dropout(dropout)
        self.linear2_4 = nn.Linear(d_ffn, d_model)
        self.dropout3_4 = nn.Dropout(dropout)
        self.norm2_4 = nn.LayerNorm(d_model)
        self.config = config
        self.gate = TCSLayer(d_model,config)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_ffn_2(self, src):
        src2 = self.linear2_3(self.dropout2_3(self.activation_3(self.linear1_3(src))))
        src = src + self.dropout3_3(src2)
        src = self.norm2_3(src)
        return src


    def forward(self, query, query_pos, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, 
                spatial_mask=None,query_index=None,layer_index=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points[:,:32400], src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        gate, gate_det,gate_seg = None, None, None
        # ffn
        num_seg_query = self.config.get('SEG_QUERY_NUM')+200
        src1 = self.forward_ffn(src)

        src = src1
        if self.config.get('TCS_WITH_CA',False):

            gate_det,gate_seg,gate_occ = self.gate(src)
            src_det,src_seg = src*gate_det, src*gate_seg
            query2_det = self.cross_attn(self.with_pos_embed(query[:,:200], query_pos[:,:200]), 
                                    reference_points[:,32400:32600], src_det, 
                                    spatial_shapes, level_start_index, padding_mask)
            query2_seg = self.cross_attn(self.with_pos_embed(query[:,200:num_seg_query], query_pos[:,200:num_seg_query]), 
                                    reference_points[:,32600:32400+num_seg_query], src_seg, 
                                    spatial_shapes, level_start_index, padding_mask)

            if self.config.get('TRAIN_OCC',False):
                src_occ = src*gate_occ
                query2_occ = self.cross_attn(self.with_pos_embed(query[:,num_seg_query:], query_pos[:,num_seg_query:]), 
                                        reference_points[:,32400+num_seg_query:], src_occ, 
                                        spatial_shapes, level_start_index, padding_mask)
                query2 = torch.cat([query2_det,query2_seg,query2_occ],1)
            else:
                query2 = torch.cat([query2_det,query2_seg],1)
        else:
            src_index = src[:,query_index]
            gate_det,gate_seg,gate_occ = self.gate(src_index)
            query_det = src[:,query_index[:200]]
            query_seg = src[:,query_index[200:num_seg_query]]
            if self.config.get('TRAIN_OCC',False):
                query_occ = src[:,query_index[num_seg_query:]]
                query_occ = query_occ*gate_occ[:,num_seg_query:]
                query_det,query_seg = query_det*gate_det[:,:200], query_seg*gate_seg[:,200:num_seg_query]
                query2 = torch.cat([query_det,query_seg,query_occ],1)
            else:
                query_det,query_seg = query_det*gate_det[:,:200], query_seg*gate_seg[:,200:num_seg_query]
                query2 = torch.cat([query_det,query_seg],1)

        query = query + self.dropout_query(query2)
        query = self.norm_query(query)
        query = self.forward_ffn_2(query)

        return query, src, gate, gate_det, gate_seg

class TransformerDecoderDet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        mask_score=0.5,
        down_sample=4,
        config = None

    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.config = config
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.mask_score = mask_score
        self.down_sample = down_sample
        self.only_train_line = config.ONLY_TRAIN_LINE
        self.preslice = config.MASK_FORMER.PRE_SLICE
        self.use_sincos_embed = config.MASK_FORMER.get('USE_SINCOS_EMB',False)

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.num_feature_levels = 1
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, 10)
            if self.config.TRAIN_MAP:
                self.line_class_embed = nn.Linear(hidden_dim, 6)
            else:
                self.line_class_embed = nn.Linear(hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.line_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.vel_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        if self.down_sample!=8:
            self.refine = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.grid_conf = {
            'xbound': [-54, 54, 0.6],
            'ybound': [-54, 54, 0.6],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        if config.MASK_FORMER.get('FULL_MAP',False):
            map_grid_conf = {
                'ybound': [-50.0, 50.0, 0.25],
                'xbound': [-50.0, 50.0, 0.25],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }
        else:
            map_grid_conf = {
                'ybound': [-30.0, 30.0, 0.15],
                'xbound': [-15.0, 15.0, 0.15],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }

        motion_grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        # self.lineslicer = BevFeatureSlicer(grid_conf=motion_grid_conf,map_grid_conf=map_grid_conf)
        self.train_vel = config.MASK_FORMER.get('TRAIN_VEL', False)
        self.train_line = config.MASK_FORMER.get('TRAIN_LINE', False)
        self.attnslicer = BevFeatureSlicer(grid_conf=motion_grid_conf,map_grid_conf=self.grid_conf)


    def forward(self, x, mask_features, line_features, mask, query_feat, query_embed, bev_embed, weight_mask=None,weight_line=None,boxes=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # import pdb;pdb.set_trace()
        # disable mask, it does not affect performance
        del mask

        if self.use_sincos_embed:
            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

        else:
            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(bev_embed.flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = query_embed.permute(2,0,1).contiguous()
        output = query_feat.permute(2,0,1).contiguous()

        predictions_query = []

        attn_mask = None
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if not attn_mask is None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, 
                                                               tgt_key_padding_mask=None,query_pos=query_embed)

            output = self.transformer_ffn_layers[i](
                    output
                )

            predictions_query.append(output.permute(1,2,0))

        out = {
            'query_feat': predictions_query,
        }

        return out,None


    def forward_prediction_heads(self, output, mask_features,line_features,attn_mask_target_size, query_emb, bev_emb,score=0.5,attn_mask=None):

        if self.only_train_line:
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = None #self.class_embed(decoder_output)
            outputs_lines_cls = self.line_class_embed(decoder_output[:,-20:])
            # mask_embed = self.mask_embed(decoder_output)
            line_embed = self.line_embed(decoder_output)

            if not self.preslice:
                line_features = self.lineslicer(line_features).transpose(2,3)
                # feat = self.maskslicer(line_features).transpose(2,3)
            outputs_line = torch.einsum("bqc,bchw->bqhw", line_embed, line_features)
            outputs_mask_8x = None
            # outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed[:,:200], feat_mask)
            # import matplotlib.pyplot as plt;plt.imshow(outputs_line[0,0].cpu().numpy(),cmap='jet');plt.show()
            # outputs_line_4x = torch.einsum("bqc,bchw->bqhw", line_embed[:,200:], mask_features_4x)
            # outputs_mask_4x = torch.cat([outputs_mask_4x,outputs_line_4x],1)
            # outputs_mask_4x = None
            # outputs_mask_4x = torch.einsum("bqc,bchw->bqhw", mask_embed+query_emb, mask_features_4x+bev_emb.reshape(*mask_features.shape))
            # outputs_mask_4x = torch.einsum("bqc,bchw->bqhw", torch.cat([mask_embed,query_emb],-1), 
            #                                torch.cat([mask_features_4x,bev_emb.reshape(*mask_features.shape)],1))
            
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            # attn_mask = F.interpolate(outputs_mask_4x, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # line_mask = outputs_mask_8x[:,-1:].repeat(1,line_embed.shape[1],1,1)
            # attn_mask = outputs_line #torch.cat([outputs_mask_8x,line_mask],1)
            # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < score).bool()
            # attn_mask = attn_mask.detach()
            attn_mask =None
            vel_embed = None

        else:
            assert output.shape[0] == 220
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output[:,:200])
            outputs_lines_cls = self.line_class_embed(decoder_output[:,-20:])
            mask_embed = self.mask_embed(decoder_output[:,:200])
            if self.train_vel:
                vel_embed = self.vel_embed(decoder_output)
            else:
                vel_embed = None

            outputs_mask_8x = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            if self.train_line:
                line_embed = self.line_embed(decoder_output)[:,-20:]
                if not self.preslice:
                    line_features = self.lineslicer(line_features).transpose(2,3)
                outputs_line = torch.einsum("bqc,bchw->bqhw", line_embed, line_features)
                # DONOT use mask for line
                line_mask = torch.ones_like(outputs_mask_8x[:,-1:].repeat(1,line_embed.shape[1],1,1))
                attn_mask = torch.cat([outputs_mask_8x,line_mask],1)
            else:
                outputs_line = None
                attn_mask = outputs_mask_8x

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            # import pdb;pdb.set_trace()
            attn_mask = self.attnslicer(attn_mask).transpose(2,3)
            # attn_mask = F.interpolate(attn_mask, size=(180,180), mode="bilinear", align_corners=False)
            # line_mask = torch.ones_like(outputs_mask_8x[:,-1:].repeat(1,line_embed.shape[1],1,1))
            # attn_mask = torch.cat([outputs_mask_8x,line_mask],1)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < score).bool()
            if self.config.get('NOT_SHARE_DET_QUERY', False):
                # import pdb;pdb.set_trace()
                if self.config.get('NOT_USE_DET_MASK', True):
                    boxes_mask = torch.zeros(attn_mask.shape[0],200,180*180).cuda().bool()
                    attn_mask = torch.cat([boxes_mask,attn_mask],1)
            attn_mask = attn_mask.detach()

        return outputs_class, outputs_lines_cls, outputs_mask_8x, outputs_line, vel_embed, attn_mask 

class TransformerDecoderMultiTask(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        mask_score=0.5,
        down_sample=4,
        config = None

    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # self.pe_layer = PositionEmbeddingLearned(2, hidden_dim)
        
        # define Transformer decoder here
        self.config = config
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.mask_score = mask_score
        self.down_sample = down_sample
        self.hidden_dim = hidden_dim
        self.use_sincos_embed = config.MASK_FORMER.get('USE_SINCOS_EMB',False)
        self.num_queries = num_queries


        for _ in range(self.num_layers):

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            # if self.config.get('USE_TASK_ATTN',False) or self.config.get('USE_TASK_ATTN_V2',False):
            #     self.transformer_self_attention_layers_det.append(
            #         SelfAttentionLayer(
            #             d_model=hidden_dim,
            #             nhead=nheads,
            #             dropout=0.0,
            #             normalize_before=pre_norm,
            #         )
            #     )
            #     self.transformer_self_attention_layers_map.append(
            #         SelfAttentionLayer(
            #             d_model=hidden_dim,
            #             nhead=nheads,
            #             dropout=0.0,
            #             normalize_before=pre_norm,
            #         )
            #     )

            # if self.config.get('MAP_CLS_DIST_ATTN_AFTER_CROSS', False) or self.config.get('MAP_CLS_DIST_ATTN_BEFORE_CROSS', False):
            #     self.transformer_self_attention_layers_cls.append(
            #         SelfAttentionLayer(
            #             d_model=hidden_dim,
            #             nhead=nheads,
            #             dropout=0.0,
            #             normalize_before=pre_norm,
            #         )
            #     )
            #     self.transformer_self_attention_layers_dist.append(
            #         SelfAttentionLayer(
            #             d_model=hidden_dim,
            #             nhead=nheads,
            #             dropout=0.0,
            #             normalize_before=pre_norm,
            #         )
            #     )

            # if self.config.get('MAP_USE_DEFORM',False):
            #     self.transformer_cross_attention_layers.append(
            #         MSDeformAttnTransformerDecoderLayer(
            #                 d_model=hidden_dim,
            #                 n_levels=1, n_heads=8, n_points=8
            #             )
            #     )
            # else:
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )


        self.transformer_ffn_layers_line = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_ffn_layers_line.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_feature_levels = 1
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, 10)
            if self.config.TRAIN_MAP:
                self.line_class_embed = nn.Linear(hidden_dim, 6)
            else:
                self.line_class_embed = nn.Linear(hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.line_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.vel_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        if self.down_sample!=8:
            self.refine = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.grid_conf = {
            'xbound': [-54, 54, 0.6],
            'ybound': [-54, 54, 0.6],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        if config.MASK_FORMER.get('FULL_MAP',False):
            map_grid_conf = {
                'ybound': [-50.0, 50.0, 0.25],
                'xbound': [-50.0, 50.0, 0.25],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }
        else:
            map_grid_conf = {
                'ybound': [-30.0, 30.0, 0.15],
                'xbound': [-15.0, 15.0, 0.15],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],
            }

        motion_grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        self.train_vel = config.MASK_FORMER.get('TRAIN_VEL', False)
        self.train_line = config.MASK_FORMER.get('TRAIN_LINE', False)
        self.attnslicer = BevFeatureSlicer(grid_conf=motion_grid_conf,map_grid_conf=self.grid_conf)


    def forward(self, x, mask_features, line_features, mask, query_feat, query_embed, bev_embed, weight_mask=None,weight_line=None,anchors=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # disable mask, it does not affect performance
        del mask
        # import pdb;pdb.set_trace()
        if self.use_sincos_embed:
            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

        else:
            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(bev_embed.flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = query_embed.permute(2,0,1).contiguous()
        output = query_feat.permute(2,0,1).contiguous()

        predictions_class = []
        predictions_mask = []
        predictions_query = []
        predictions_class_line = []
        predictions_line = None
        predictions_vels = None
        if self.train_line:
            predictions_line = []
        if self.train_vel:
            predictions_vels = []

        q_emb = None
        b_emb = None 
        mask_output = output[self.num_queries:]
    
        attn_mask = None
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if not attn_mask is None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            tgt_mask=None
            output = self.transformer_self_attention_layers[i](output, tgt_mask=tgt_mask,tgt_key_padding_mask=None,query_pos=query_embed)
            output_box = self.transformer_ffn_layers[i](output[:self.num_queries])
            output_line = self.transformer_ffn_layers_line[i](output[self.num_queries:])
            output = torch.cat([output_box,output_line],dim=0)

            if self.config.get('ONLY_TRAIN_MAP',False):
                mask_output = output
            else:
                mask_output = output[self.num_queries:]

            outputs_class, outputs_lines_cls, outputs_mask, outputs_line, outputs_vels, _ = self.forward_prediction_heads(mask_output, 
                            mask_features, line_features,attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],query_emb=q_emb,bev_emb=b_emb)

            predictions_query.append(output.permute(1,2,0))
            predictions_class.append(outputs_class)
            predictions_class_line.append(outputs_lines_cls)
            predictions_mask.append(outputs_mask)
            if self.train_line:
                predictions_line.append(outputs_line)
            if self.train_vel:
                predictions_vels.append(outputs_vels)

        out_ori = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_lines': predictions_line[-1] if predictions_line  is not None else None,
            'pred_vels': predictions_vels[-1] if predictions_vels is not None else None,
            'aux_outputs': self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask, predictions_line, predictions_vels)
        }

        out = {
            'query_feat': predictions_query,
            'pred_logits': predictions_class,
            'pred_logits_line': predictions_class_line,
            'pred_masks': predictions_mask,
            'pred_lines': predictions_line,
        }

        return out,out_ori

    def forward_prediction_heads(self, output, mask_features,line_features,attn_mask_target_size, query_emb, bev_emb,score=0.5,attn_mask=None,idx=None):

        outputs_class = None
        outputs_mask = None

        if idx ==0:
            line_embed = output.transpose(0, 1)
        else:
            output = self.decoder_norm(output)
            output = output.transpose(0, 1)
            line_embed = self.line_embed(output)

        outputs_lines_cls = None
        mask_embed = None
        vel_embed = None

        line_embed = self.line_embed(output)

        num_seg_query = self.config.get('SEG_QUERY_NUM',6)

        if self.config.get('USE_ONE_SEGFEAT',False):
            b,c,h,w = line_features.shape
            if num_seg_query==6:
                outputs_line = torch.einsum("bqc,bchw->bqhw", line_embed.reshape(-1,1,line_embed.shape[-1]), line_features).permute(1,0,2,3).contiguous()
            else:
                split = num_seg_query//6
                line_feats = line_features#.reshape(b,c,split,h//split,w)
                line = line_embed.reshape(split,6,line_embed.shape[-1])
                outputs_line_list = []
                for i in range(split):
                    outputs_line = torch.einsum("bqc,bchw->bqhw", line[i:i+1], line_feats)
                    outputs_line = outputs_line.reshape(b,6,split,h//split,w)
                    outputs_line_list.append(outputs_line[:,:,i])
                outputs_line = torch.cat(outputs_line_list,2)

        attn_mask = None

        return outputs_class, outputs_lines_cls, outputs_mask, outputs_line, vel_embed, attn_mask 

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_lines=None, outputs_vels=None):

        if self.mask_classification:
            if outputs_lines is not None:
                if outputs_vels is not None:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_lines": c, "pred_vels": d}
                        for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1],outputs_lines[:-1], outputs_vels[:-1])
                    ]
                else:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_lines": c,"pred_vels": None}
                        for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],outputs_lines[:-1])
                    ]
            else:
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_lines": None,"pred_vels": None}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

