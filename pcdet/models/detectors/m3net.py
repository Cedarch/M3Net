from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser, neck
from .. import dense_heads
try:
    from tools.visual_utils import open3d_vis_utils as V
except:
    pass

class M3Net(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'backbone_2d_neck', 'dense_head',  'point_head', 'roi_head'
        ]

        self.module_list = self.build_networks()
       
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict

    def build_backbone_2d_neck(self,model_info_dict):
        if self.model_cfg.get('BACKBONE_2D_NECK', None) is None:
            return None, model_info_dict
        
        backbone_2d_neck_module = neck.__all__[self.model_cfg.BACKBONE_2D_NECK.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D_NECK
        )
        model_info_dict['module_list'].append(backbone_2d_neck_module)
        model_info_dict['num_bev_features'] = self.model_cfg.BACKBONE_2D_NECK.out_channels
        return backbone_2d_neck_module, model_info_dict

    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor,
            'receptive_field': None,
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']
    
    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False),
            receptive_field = model_info_dict.get('receptive_field',None)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def forward(self, batch_dict):

        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # import pdb;pdb.set_trace()
            if self.model_cfg.DENSE_HEAD.get('EVAL_DET',False):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                if 'pred_results' in batch_dict.keys():
                    pred_dicts[0]['pred_results'] = batch_dict['pred_results']
                if 'map_results' in batch_dict.keys():
                    pred_dicts[0]['map_results'] = batch_dict['map_results']
                if 'occ_results' in batch_dict.keys():
                    pred_dicts[0]['occ_results'] = batch_dict['occ_results']
                return pred_dicts, recall_dicts
            elif self.model_cfg.DENSE_HEAD.get('ONLY_EVAL_MAP',False):
                pred_dicts = [{
                    'masks_bev': batch_dict['masks_bev'],
                    'gt_masks_bev': batch_dict['gt_masks_bev']}]
                return pred_dicts
            else:
                pred_dicts = batch_dict['pred_results']
                recall_dicts = None
                return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans = batch_dict['loss'] #,batch_dict['tb_dict']
        tb_dict = batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = len(batch_dict['final_box_dicts'])
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
