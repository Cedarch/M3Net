import copy
import pickle
from pathlib import Path
import torch
import numpy as np
import copy
from tqdm import tqdm
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image
from nuscenes.map_expansion.map_api import NuScenesMap
from .nuscenes_utils import VectorizedLocalMap, preprocess_map, cm_to_ious,format_SC_results,format_SSC_results
try:
    from tools.visual_utils import open3d_vis_utils as V
except:
    pass

class NuScenesDatasetMTL(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.map_config = self.dataset_cfg.get('MAP_CONFIG',None)
        if self.map_config is not None:
            self.use_map = self.map_config.get('USE_MAP',True)
            self.map_classes = self.map_config.CLASS_NAMES
        else:
            self.use_map = False
        #import pdb;pdb.set_trace()
        self.include_nuscenes_data(self.mode)
        # self.nusc_maps = self.load_all_maps(root_path)
        #import pdb;pdb.set_trace()
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

        # import pdb;pdb.set_trace()
        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            # sampled_waymo_infos = []
            # for k in range(0, len(nuscenes_infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
            #     sampled_waymo_infos.append(nuscenes_infos[k])
            self.infos = self.infos[::self.dataset_cfg.SAMPLED_INTERVAL[self.mode]]
            self.logger.info('Total sampled samples for Nuscenes dataset: %d' % len(self.infos))


        if self.dataset_cfg.get('ONLY_TRAIN_TRAILER',False):
            class_names =['motocycle', 'bicycle']
            cls_infos = {name: [] for name in class_names}
            for info in self.infos:
                for name in set(info['gt_names']):
                    if name in class_names:
                        if info['scene_token'] not in cls_infos[name]:
                            cls_infos[name].append(info['scene_token'])
            info_trailer_scene = []
            for info in self.infos:
                for key in class_names:
                    if info['scene_token'] in cls_infos[key]: #info['scene_token'] in cls_infos['bus'] or info['scene_token'] in cls_infos['bicycle']:
                        info_trailer_scene.append(info)
            #trailer_scene = ['9d1307e95c524ca4a51e03087bd57c29','7bd098ac88cb4221addd19202a7ea5de']
                            #  'd7bacba9119840f78f3c804134ceece0','7210f928860043b5a7e0d3dd4b3e80ff'
                            #  '19d97841d6f64eba9f6eb9b6e8c257dc']
            # import pdb;pdb.set_trace()
            #bus
            # trailer_scene = ['16e50a63b809463099cb4c378fe0641e','55b3a17359014f398b6bbd90e94a8e1b']
                            #  '905cfed4f0fc46679e8df8890cca4141','04219bfdc9004ba2af16d3079ecc4353']
            # bicycle
            # trailer_scene =['afd73f70ff7d46d6b772d341c08e31a5','9f1f69646d644e35be4fe0122a8b91ef','c525507ee2ef4c6d8bb64b0e0cf0dd32']
            # # moto
            # trailer_scene = ['325cef682f064c55a255f2625c533b75','aacd6706a091407fb1b0e5343d27da7e','201b7c65a61f4bc1a2333ea90ba9a932']
            # truck 
            # trailer_scene = ['7210f928860043b5a7e0d3dd4b3e80ff', '7bd098ac88cb4221addd19202a7ea5de', '991d65cab952449a821deb32e971ff19']
            """
            for info in self.infos:
                if info['scene_token'] in trailer_scene:
                    info_trailer_scene.append(info)
            """
            self.infos = info_trailer_scene
        
        """
        grid_conf = {
            'ybound': [-30.0, 30.0, 0.15],
            'xbound': [-15.0, 15.0, 0.15],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }

        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)
        self.road_canvas_size = (int(50*2/0.5), int(50*2/0.5))

        self.map_max_channel=3
        self.map_thickness=5
        self.map_angle_class=36
        # self.downsample=dataset_cfg.DOWN_SAMPLE
        self.fusion = dataset_cfg.TEMP_FUSION
        self.load_road_map = dataset_cfg.LOAD_ROAD_MAP
        self.use_future_feat = dataset_cfg.get('USE_FUTURE_FEAT', False)
        self.receptive_field = -dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0] + 1

        self.vector_map = VectorizedLocalMap(
            dataroot=root_path,
            patch_size=self.map_patch_size,
            canvas_size=self.map_canvas_size,
            canvas_size_road=self.road_canvas_size,
            patch_size_road = (108,108)
        )
        """
        grid_conf = {
            'ybound': [-33.0, 33.0, 0.15],
            'xbound': [-33.0, 33.0, 0.15],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 60.0, 1.0],
        }
        
        map_xbound, map_ybound = grid_conf['xbound'], grid_conf['ybound']
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])
        if dataset_cfg.get('FULL_MAP', False):
            self.map_patch_size = (100, 100)
            self.map_canvas_size = (int(50*2/0.25), int(50*2/0.25))
            self.map_thickness=3
        else:
            self.map_patch_size = (patch_h, patch_w)
            self.map_canvas_size = (canvas_h, canvas_w)
            self.map_thickness=5
        self.road_canvas_size = (int(50*2/0.5), int(50*2/0.5))

        self.map_max_channel=3
        
        self.map_angle_class=36
        self.downsample=dataset_cfg.DOWN_SAMPLE
        try:
            self.temp_fusion = dataset_cfg.TEMP_FUSION
        except:
            self.temp_fusion = dataset_cfg.FUSION
        self.fusion = self.temp_fusion
        self.load_road_map = dataset_cfg.LOAD_ROAD_MAP
        self.train_future = dataset_cfg.get('TRAIN_FUTURE', False)
        self.use_future_feat = dataset_cfg.get('USE_FUTURE_FEAT', False)
        self.receptive_field = -dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0] + 1

        self.vector_map = VectorizedLocalMap(
            dataroot=root_path,
            patch_size=self.map_patch_size,
            canvas_size=self.map_canvas_size,
            canvas_size_road=self.road_canvas_size,
            patch_size_road = (100,100)
        )

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []
        # import pdb;pdb.set_trace()
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)
        self.infos = nuscenes_infos
        # if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
        #     sampled_waymo_infos = []
        #     for k in range(0, len(nuscenes_infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
        #         sampled_waymo_infos.append(nuscenes_infos[k])
        #     self.infos = sampled_waymo_infos
        #     self.logger.info('Total sampled samples for Nuscenes dataset: %d' % len(self.infos))
        # else:
        #     self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def reorganize_info_by_token(self, infos):
        """
        Reorganize the info dict by token.
        """
        info_by_token = {}
        for k, info in enumerate(infos):
            if info['token'] not in info_by_token:
                info_by_token[info['token']] = k
        return info_by_token

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos
        self.order_infos = infos
        self.token2order = self.reorganize_info_by_token(infos)
        if self.dataset_cfg.get('ONLY_TRAIN_TRAILER',False):
            class_names = ['trailer','bus', 'bicycle']
        else:
            class_names = self.class_names
        cls_infos = {name: [] for name in class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1,cbgs=False):
        if cbgs:
            info = self.order_infos[index]
        else:
            info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        input_dict['ori_imgs'] = [np.array(img) for img in imgs]
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info_sequence(self, input_dict, info, sample_idx, sequence_cfg):

        input_dict['image_sequence'] = []
        if self.training or self.fusion:
            if self.use_future_feat:
                offset_list = np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1])
            else:
                offset_list = np.arange(sequence_cfg.SAMPLE_OFFSET[0], 1)
            if self.dataset_cfg.get('BALANCED_RESAMPLING', False):
                order_index = self.token2order[info['token']]
                assert self.order_infos[order_index]['token'] == info['token']
                sample_idx_pre_list = np.clip(order_index + offset_list, 0, 28129)
            else:
                sample_idx_pre_list = np.clip(sample_idx + offset_list, 0, 28129)
        else:
            sample_idx_pre_list = [sample_idx]

        for index in sample_idx_pre_list:
            image_seq_dict = {}
            image_seq_dict["image_paths"] = []
            image_seq_dict["lidar2camera"] = []
            image_seq_dict["lidar2image"] = []
            image_seq_dict["camera2ego"] = []
            image_seq_dict["camera_intrinsics"] = []
            image_seq_dict["camera2lidar"] = []
            # info = copy.deepcopy(self.infos[index])
            if hasattr(self, 'order_infos'):
                info = copy.deepcopy(self.order_infos[index])
            else:
                info = copy.deepcopy(self.infos[index])
            for _, camera_info in info["cams"].items():
                image_seq_dict["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                image_seq_dict["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                image_seq_dict["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                image_seq_dict["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                image_seq_dict["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                image_seq_dict["camera2lidar"].append(camera2lidar)
            # read image
            filename = image_seq_dict["image_paths"]
            images = []
            for name in filename:
                images.append(Image.open(str(self.root_path / name)))
            
            image_seq_dict["camera_imgs"] = images
            image_seq_dict["ori_shape"] = images[0].size
            # resize and crop image
            
            image_seq_dict = self.crop_image(image_seq_dict)
            input_dict['image_sequence'].append(image_seq_dict)

        return input_dict

    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def transform_prebox_to_current_numpy(self, pred_boxes3d, pose_pre, pose_cur):

        pred_boxes3d = pred_boxes3d.copy()
        expand_bboxes = np.concatenate([pred_boxes3d[:, :3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)

        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        expand_bboxes_global = np.concatenate([bboxes_global[:, :3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        pred_boxes3d[:, 0:3] = bboxes_pre2cur

        if pred_boxes3d.shape[-1] == 11:
            expand_vels = np.concatenate([pred_boxes3d[:, 7:9], np.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
            vels_global = np.dot(expand_vels, pose_pre[:3, :3].T)
            vels_pre2cur = np.dot(vels_global, np.linalg.inv(pose_cur[:3, :3].T))[:,:2]
            pred_boxes3d[:, 7:9] = vels_pre2cur

        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] + np.arctan2(pose_pre[..., 1, 0], pose_pre[..., 0, 0])
        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] - np.arctan2(pose_cur[..., 1, 0], pose_cur[..., 0, 0])
        return pred_boxes3d

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, gt_names, gt_boxes, gt_ids):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """

        if self.training or self.fusion:
            offset_list = np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1])
            if self.dataset_cfg.get('BALANCED_RESAMPLING', False):
                order_index = self.token2order[info['token']]
                assert self.order_infos[order_index]['token'] == info['token']
                sample_idx_pre_list = np.clip(order_index + offset_list, 0, 0x7FFFFFFF)
                sample_idx_pre_list = sample_idx_pre_list[::-1]
            else:
                sample_idx_pre_list = np.clip(sample_idx + offset_list, 0, 0x7FFFFFFF)
                sample_idx_pre_list = sample_idx_pre_list[::-1]
        else:
            sample_idx_pre_list = []
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        points_pre_all = [points]
        gt_boxes = np.hstack([gt_boxes, np.zeros((gt_boxes.shape[0], 1)).astype(gt_boxes.dtype)])
        gt_boxes_all = [gt_boxes]
        gt_names_all = [gt_names]
        gt_ids_all = [gt_ids]
        num_points_all = [info['num_lidar_pts']]
        # ref_from_car_list = [info['ref_from_car']]
        # car_from_global_list = [info['car_from_global']]
        vector_list = []
        # gt_boxes_all = [np.zeros([0,10])]
        # gt_names_all = [np.zeros(0)]
        # gt_ids_all = [np.zeros(0)]
        # num_points_all = [np.zeros(0)]
        # import pdb;pdb.set_trace()
        if self.load_road_map and self.training:
            center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), info['global_from_ref'].reshape((4, 4)).T)[0]
            angle = np.arctan2(info['global_from_ref'].reshape((4, 4))[..., 1, 0], info['global_from_ref'].reshape((4, 4))[..., 0,0])/np.pi *180
            if len(self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST) <= 2:
                map_results = self.get_map_ann_info(info,center_global,angle)
                semantic_map_mask = map_results['semantic_indices']
                map_mask_8x_all = [map_results['road_map'][None]]
                semantic_map_4x_all = [semantic_map_mask]
            else:
                vectors, map_mask = self.vector_map.gen_vectorized_samples(
                    info['map_name'], center_global,angle)
                for vector in vectors:
                    pts = vector['pts']
                    vector['pts'] = np.concatenate(
                        (pts, np.zeros((pts.shape[0], 1))), axis=1)
                map_mask_8x_all = [None]
                semantic_map_4x_all = [None]
                vector_list.append(vectors)
        else:
            pass
            # map_mask_8x_all = [info['road_mask'][None]]
            # semantic_map_4x_all = [info['line_mask']]
            # instance_mask_all = [info['instance_mask']]
            # instance_map_all = [info['instance_map']]

        # ref_from_global_list = [info['ref_from_global'].reshape((4, 4))]
        global_from_ref_list = [info['global_from_ref'].reshape((4, 4))]
        ref_from_car_list = [info['ref_from_car'].reshape((4, 4))]
        car_from_global_list = [info['car_from_global'].reshape((4, 4))]
        timestamp = info['timestamp']
        timelag_list = [0]
        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):
            if hasattr(self, 'order_infos'):
                pre_info = copy.deepcopy(self.order_infos[sample_idx_pre])
            else:
                pre_info = copy.deepcopy(self.infos[sample_idx_pre])
            pre_sequence_name = pre_info['scene_token']

            if pre_sequence_name == sequence_name:
                CBGS = self.dataset_cfg.get('BALANCED_RESAMPLING', False) or self.dataset_cfg.get('BALANCED_RESAMPLING_MAP', False)
                pre_points = self.get_lidar_with_sweeps(sample_idx_pre, self.dataset_cfg.MAX_SWEEPS,CBGS)
                # expand_points_pre = np.concatenate([pre_points[:, :3], np.ones((pre_points.shape[0], 1))], axis=-1)
                # points_pre_global = np.dot(expand_points_pre, pre_info['global_from_ref'].reshape((4, 4)))[:, :3]
                # expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
                # points_pre2cur = np.dot(expand_points_pre_global, info['ref_from_global'].reshape((4, 4)))[:, :3]
                # pre_points = np.concatenate([points_pre2cur, pre_points[:, 3:]], axis=-1)
                # import pdb;pdb.set_trace()
                # ref_from_global_list.append(pre_info['ref_from_global'].reshape((4, 4)))
                global_from_ref_list.append(pre_info['global_from_ref'].reshape((4, 4)))
                ref_from_car_list.append(pre_info['ref_from_car'].reshape((4, 4)))
                car_from_global_list.append(pre_info['car_from_global'].reshape((4, 4)))

                pre_points = np.hstack([pre_points,  (idx+1) * np.ones((pre_points.shape[0], 1)).astype(pre_points.dtype)])  # one frame 0.1s
                if 'gt_boxes' in pre_info:
                    if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                        mask = (pre_info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                    else:
                        mask = None

                    gt_names = pre_info['gt_names'] if mask is None else pre_info['gt_names'][mask]
                    gt_boxes = pre_info['gt_boxes'] if mask is None else pre_info['gt_boxes'][mask]
                    gt_ids   = pre_info['instance_inds'] if mask is None else pre_info['instance_inds'][mask]
                    gt_boxes = np.hstack([gt_boxes,  (idx+1) * np.ones((gt_boxes.shape[0], 1)).astype(gt_boxes.dtype)])
                    gt_names_all.append(gt_names)
                    gt_boxes_all.append(gt_boxes)
                    gt_ids_all.append(gt_ids)
                    num_points_all.append(pre_info['num_lidar_pts'])
                # import pdb;pdb.set_trace()
                # trans_gt = self.transform_prebox_to_current_numpy(gt_boxes, pre_info['global_from_ref'].reshape((4, 4)), info['global_from_ref'].reshape(4, 4))
                

                points_pre_all.append(pre_points)
                timelag = timestamp - pre_info['timestamp']
                timelag_list.append(timelag)
                timestamp = pre_info['timestamp']

                if self.load_road_map and self.training:
                    center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), pre_info['global_from_ref'].reshape((4, 4)).T)[0]
                    angle = np.arctan2(pre_info['global_from_ref'].reshape((4, 4))[..., 1, 0], 
                                        pre_info['global_from_ref'].reshape((4, 4))[..., 0, 0])/np.pi *180
                    if len(self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST) <= 2:
                        map_results = self.get_map_ann_info(pre_info,center_global,angle)
                        semantic_map_mask = map_results['semantic_indices']
                        semantic_map_4x_all.append(semantic_map_mask)
                        map_mask_8x_all.append(map_results['road_map'][None])
                    else:
                        vectors, map_mask = self.vector_map.gen_vectorized_samples(
                            info['map_name'], center_global,angle)
                        for vector in vectors:
                            pts = vector['pts']
                            vector['pts'] = np.concatenate(
                                (pts, np.zeros((pts.shape[0], 1))), axis=1)
                        semantic_map_4x_all.append(None)
                        map_mask_8x_all.append(None)
                        vector_list.append(vectors)
                else:
                    pass
                    # map_mask_8x_all.append(pre_info['road_mask'][None])
                    # semantic_map_4x_all.append(pre_info['line_mask'])
                    # instance_mask_all.append(pre_info['instance_mask'])
                
            else:
                # import pdb;pdb.set_trace()
                pre_points = points_pre_all[-1][:,:5]
                pre_points = np.hstack([pre_points,  (idx+1) * np.ones((pre_points.shape[0], 1)).astype(pre_points.dtype)])
                gt_boxes = gt_boxes_all[-1][:,:9]
                gt_boxes = np.hstack([gt_boxes,  (idx+1) * np.ones((gt_boxes.shape[0], 1)).astype(gt_boxes.dtype)])
                points_pre_all.append(pre_points)
                gt_names_all.append(gt_names_all[-1])
                gt_ids_all.append(gt_ids_all[-1])
                gt_boxes_all.append(gt_boxes)
                # ref_from_global_list.append(ref_from_global_list[-1])
                global_from_ref_list.append(global_from_ref_list[-1])
                ref_from_car_list.append(ref_from_car_list[-1])
                car_from_global_list.append(car_from_global_list[-1])
                timelag_list.append(0)
                # map_mask_8x_all.append(map_mask_8x_all[-1])
                # instance_map_all.append(instance_map_all[-1])
                # semantic_map_4x_all.append(semantic_map_4x_all[-1])
                num_points_all.append(num_points_all[-1])
                # if self.load_road_map:
                #     vector_list.append(vector_list[-1])
                # if not self.load_road_map:
                #     instance_mask_all.append(instance_mask_all[-1])


        points = np.concatenate(points_pre_all, axis=0).astype(np.float32) 
        gt_names = np.concatenate(gt_names_all[::-1], axis=0)
        gt_boxes = np.concatenate(gt_boxes_all[::-1], axis=0)
        num_points = np.concatenate(num_points_all[::-1], axis=0)
        gt_ids = np.concatenate(gt_ids_all[::-1], axis=0)
        if self.load_road_map and self.training:
            if len(self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST) <= 2:
                vectors = None
                map_mask_8x = np.concatenate(map_mask_8x_all[::-1], axis=0)
                semantic_map_4x = np.stack(semantic_map_4x_all[::-1],0)
            else:
                vectors = vector_list[::-1]
                map_mask_8x = None
                semantic_map_4x = None
        else:
            vectors = None
            map_mask_8x = None # np.concatenate(map_mask_8x_all[::-1], axis=0)
            semantic_map_4x = None # np.stack(semantic_map_4x_all[::-1],0)
    
        instance_map = None
        # if not self.load_road_map or not self.training:
        #     instance_mask = np.stack(instance_mask_all[::-1],0)
        # else:
        instance_mask = None
        ref_from_global = None #np.stack(ref_from_global_list[::-1],0)
        ref_from_car = np.stack(ref_from_car_list[::-1],0)
        car_from_global = np.stack(car_from_global_list[::-1],0)
        if not self.training:
            future_idx = np.clip(sample_idx+1,0,len(self.infos)-1)
            future_info = copy.deepcopy(self.infos[future_idx])
            global_from_ref_future = future_info['global_from_ref'].reshape((4, 4))
            global_from_ref_list.append(global_from_ref_future)
            global_from_ref = np.stack(global_from_ref_list,0)
            timelag = future_info['timestamp'] - info['timestamp']
            timelag_list.append(timelag)
            timelag = np.stack(timelag_list,0)
        else:
            global_from_ref = np.stack(global_from_ref_list[::-1],0)
            timelag = np.stack(timelag_list[::-1],0)

        return points, gt_names, gt_boxes, ref_from_global, global_from_ref, timelag, instance_mask, map_mask_8x, semantic_map_4x, instance_map, gt_ids, num_points, vectors, ref_from_car, car_from_global
    
    def load_all_maps(self, dataroot, verbose: bool = False):
        """
        Loads all NuScenesMap instances for all available maps.
        :param helper: Instance of PredictHelper.
        :param verbose: Whether to print to stdout.
        :return: Mapping from map-name to the NuScenesMap api instance.
        """
        # dataroot = helper.data.dataroot
        maps = {}
        for map_name in locations:
            if verbose:
                print(f'static_layers.py - Loading Map: {map_name}')

            maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

        return maps

    def get_map_mask(self,patch_box,patch_angle,resolution,name):
        # patch_box = (300, 1700, 54, 54)
        # patch_angle = 0  # Default orientation where North is up
        layer_names = ['drivable_area', 'walkway']
        feat_size = int(patch_box[-1]/(0.075*resolution))
        canvas_size = (feat_size, feat_size)
        map_mask = self.nusc_maps[name].get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        if map_mask.shape[0]==1:
            pad_ones_sidewalk = np.zero_like(map_mask)
            map_mask = np.concatenate([map_mask,pad_ones_sidewalk],0)
        # lanes = self.nusc_maps[name].get_records_in_radius(patch_box[0], patch_box[1], 54, layer_names)
        # lanes = lanes['drivable_area'] + lanes['walkway']
        # lanes = self.nusc_maps[name].discretize_lanes(lanes, discretization_meters=0.075)
        return map_mask
    
    def __len__(self):
        # if self._merge_all_iters_to_one_epoch:
        #     return len(self.infos) * self.total_epochs

        return  len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        # idx = [k for k, info in enumerate(self.infos) if info['token'] == 'd9bcbaa7609d4dccb28249df37ced535']
        # import pdb;pdb.set_trace()
        # if not self.training:
        # index = 41

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if 'lidar_token' in info:
            input_dict = {
                'points': points,
                'frame_id': Path(info['lidar_path']).stem,
                'metadata': {'token': info['token']},
                'lidar_token': info['lidar_token'],
                'scene_token': info['scene_token'],
                'timestamp': np.array(info['timestamp']),
                'is_first': info['is_first'],
                'location': info['location']
            }
        else:
            input_dict = {
                'points': points,
                'frame_id': Path(info['lidar_path']).stem,
                'metadata': {'token': info['token']},
                'scene_token': info['scene_token'],
                'timestamp': np.array(info['timestamp']),
                'is_first': info['is_first'],
                'location': info['location'] if 'location' in info else None
            }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask],
                'instance_inds': info['instance_inds'] if mask is None else info['instance_inds'][mask]
            })
        if 'gt_boxes_2d' in info:
            info['gt_boxes_2d'] = info['gt_boxes_2d'][info['empty_mask']]
            input_dict.update({
                'gt_boxes2d': info['gt_boxes_2d'] if mask is None else info['gt_boxes_2d'][mask]
            })

        sequence_name = info['scene_token']
        # if self.training:
        points, gt_names,gt_boxes,ref_from_global, global_from_ref, timelag, instance_mask, map_mask_8x, semantic_map_4x, instance_map, gt_ids, num_points,vectors,ref_from_car, car_from_global  = self.get_sequence_data(
            info, points, sequence_name, index, self.dataset_cfg.SEQUENCE_CONFIG, input_dict['gt_names'], input_dict['gt_boxes'], input_dict['instance_inds'])

        input_dict.update({
                'points': points,
                'gt_names': gt_names,
                'gt_boxes': gt_boxes[:,:9],
                'gt_boxes_index': gt_boxes[:,-1],
                'timelag': timelag,
                "ref_from_global": ref_from_global, 
                "global_from_ref": global_from_ref,
                'ref_from_car': ref_from_car, 
                'car_from_global': car_from_global,
                "map_mask_8x": map_mask_8x,
                'semantic_map':semantic_map_4x,
                # 'instance_mask':instance_mask,
                'gt_ids': gt_ids,
                'sample_idx': index,
                # 'num_points': num_points
            })
        if not self.dataset_cfg.get('ONLY_DET',False):
            input_dict['num_points'] = num_points

        if vectors is not None:
            input_dict['vectors'] = vectors
        # import pdb;pdb.set_trace()
        if self.use_camera:
            input_dict = self.load_camera_info_sequence(input_dict, info, index,self.dataset_cfg.SEQUENCE_CONFIG)
 
        # import pdb;pdb.set_trace()
        # V.draw_scenes(None,gt_boxes)
        # import matplotlib.pyplot as plt;plt.imshow(semantic_map_4x[0,0]);plt.show()

        # center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), global_from_ref[0].T)[0]
        # angle = np.arctan2(global_from_ref[..., 1, 0], global_from_ref[..., 0,0])/np.pi *180
        # map_results = self.get_map_ann_info(info,center_global,angle)
        # input_dict.update(map_results)
        # map_mask = np.zeros([3,360,360])
        # map_mask[:,180-50:180+50,180-100:180+100] = input_dict['semantic_map']
        # input_dict['semantic_map'] = map_mask

        # if 'onenorth' in info['map_name']:
        # import pdb;pdb.set_trace()
        # center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), global_from_ref[0].T)[0]
        # angle = np.arctan2(global_from_ref[..., 1, 0], global_from_ref[..., 0,0])/np.pi *180
        # map_mask_1x = self.get_map_mask(patch_box=[center_global[0],center_global[1],54*2,54*2],patch_angle=0,resolution=1,name=info['map_name'])
        # V.draw_scenes(points,gt_boxes)
        # import pdb;pdb.set_trace()
        # import matplotlib.pyplot as plt;plt.imshow(map_mask_8x[0]);plt.show()

        # input_dict.pop('instance_inds')
        # input_dict.pop('gt_boxes_index')
        # input_dict.pop('gt_ids')
        # input_dict.pop('num_points')
        # input_dict.pop('image_sequence')
        data_dict = self.prepare_data(data_dict=input_dict)


        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict
    

    def get_map_ann_info(self, info,center,angle):
        # get the annotations of HD maps
        vectors, map_mask = self.vector_map.gen_vectorized_samples(
            info['map_name'], center,angle)

        # type0 = 'divider'
        # type1 = 'pedestrain'
        # type2 = 'boundary'
        # import pdb;pdb.set_trace()
        for vector in vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1)


        for vector in vectors:
            vector['pts'] = vector['pts'][:, :2]

        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
            vectors, self.map_patch_size, self.map_canvas_size, self.map_max_channel, self.map_thickness, self.map_angle_class)

        num_cls = semantic_masks.shape[0]
        indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)

        results={
            'semantic_map': semantic_masks,
            'road_map': map_mask,
            'instance_map': instance_masks,
            'semantic_indices': semantic_indices.astype('int64'),
            'forward_direction': forward_masks,
            'backward_direction': backward_masks,
        }

        return results

    def get_line_label(self, info):
        center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), info['global_from_ref'].reshape(4,4).T)[0]
        angle = np.arctan2(info['global_from_ref'].reshape(4,4)[..., 1, 0], info['global_from_ref'].reshape(4,4)[..., 0,0])/np.pi *180
        map_results = self.get_map_ann_info(info,center_global,angle)
        map_mask = np.zeros([3,180,180])
        map_mask[:,90-50:90+50,90-25:90+25] = map_results['semantic_map']
        return map_mask 

    def evaluation_tracking(self, det_annos, class_names, tracking=False,**kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        # if metric == 'nuscenes_tracking':
        #     nusc_annos = nuscenes_utils.transform_tracking_annos_to_nusc_annos(det_annos, nusc)
        # else:
        #     nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos, nusc_annos_det, nusc_annos_iou = nuscenes_utils.transform_tracking_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        nusc_annos_det['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        nusc_annos_iou['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        metric = kwargs['eval_metric']
        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        res_path_det = str(output_path / 'results_nusc_det_score.json')
        with open(res_path_det, 'w') as f:
            json.dump(nusc_annos_det, f)

        res_path_iou = str(output_path / 'results_nusc_iou_score.json')
        with open(res_path_iou, 'w') as f:
            json.dump(nusc_annos_iou, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.common.config import config_factory as track_configs
        from nuscenes.eval.detection.evaluate import NuScenesEval
        from nuscenes.eval.tracking.evaluate import TrackingEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }

        if metric == 'nuscenes_tracking':
            cfg = track_configs("tracking_nips_2019")

            cfg.tracking_names = [
                'bicycle',
                'bus',
                'car',
                'motorcycle',
                'pedestrian',
                'trailer',
                'truck',
                ]

            cfg.class_names =[
                'bicycle',
                'bus',
                'car',
                'motorcycle',
                'pedestrian',
                'trailer',
                'truck',
                ]

            # import pdb;pdb.set_trace()
            nusc_eval = TrackingEval(
                config=cfg,
                result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path),
                verbose=False,
                nusc_version=self.dataset_cfg.VERSION,
                nusc_dataroot='/home/xschen/repos/OpenPCDet/data/nuscenes/v1.0-trainval',


            )
            metrics_summary_track = nusc_eval.main(False)
            # with open(output_path / 'metrics_summary.json', 'r') as f:
            #     metrics = json.load(f)

            result_str, result_dict = nuscenes_utils.format_nuscene_results_tracking(metrics_summary_track, cfg.tracking_names, version="tracking_nips_2019")
        # import pdb;pdb.set_trace()
        else:
            try:
                eval_version = 'detection_cvpr_2019'
                eval_config = config_factory(eval_version)
            except:
                eval_version = 'cvpr_2019'
                eval_config = config_factory(eval_version)

            nusc_eval = NuScenesEval(
                nusc,
                config=eval_config,
                result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path),
                verbose=True,
            )
            metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

            # with open(output_path / 'metrics_summary.json', 'r') as f:
            #     metrics = json.load(f)

            result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics_summary, self.class_names, version=eval_version)
            # result_str = result_str_track + '/n' + result_str 
        return result_str, result_dict

    def evaluation(self, det_annos, class_names, tracking=False,**kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        metric = kwargs['eval_metric']
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        if metric == 'nuscenes_tracking':
            nusc_annos,_,_ = nuscenes_utils.transform_tracking_annos_to_nusc_annos(det_annos, nusc)
        else:
            nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }


        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.common.config import config_factory as track_configs
        from nuscenes.eval.detection.evaluate import NuScenesEval
        from nuscenes.eval.tracking.evaluate import TrackingEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }

        if metric == 'nuscenes_tracking':
            cfg = track_configs("tracking_nips_2019")

            cfg.tracking_names = [
                'bicycle',
                'bus',
                #'car',
                #'pedestrian',
                'motorcycle',
                'trailer',
                'truck',
                ]

            cfg.class_names =[
                'bicycle',
                'bus',
                #'car',
                #'pedestrian',
                'motorcycle',
                'trailer',
                'truck',
                ]

            # import pdb;pdb.set_trace()
            nusc_eval = TrackingEval(
                config=cfg,
                result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path),
                verbose=True,
                nusc_version=self.dataset_cfg.VERSION,
                nusc_dataroot='/home/xschen/repos/OpenPCDet/data/nuscenes/v1.0-trainval',


            )
            metrics_summary_track = nusc_eval.main(False)
            with open(output_path / 'metrics_summary.json', 'r') as f:
                metrics = json.load(f)

            result_str, result_dict = nuscenes_utils.format_nuscene_results_tracking(metrics, cfg.tracking_names, version="tracking_nips_2019")

        else:
            try:
                eval_version = 'detection_cvpr_2019'
                eval_config = config_factory(eval_version)
            except:
                eval_version = 'cvpr_2019'
                eval_config = config_factory(eval_version)
            if self.dataset_cfg.get('EVAL_DET_BY_TRACKING',False):
                eval_config.class_names =[
                    'car',
                    'truck',
                    'bus',
                    'trailer',
                    'pedestrian',
                    'motorcycle',
                    'bicycle'
                    ]

            nusc_eval = NuScenesEval(
                nusc,
                config=eval_config,
                result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path),
                verbose=True,
            )
            metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

            with open(output_path / 'metrics_summary.json', 'r') as f:
                metrics = json.load(f)

            result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, eval_config.class_names, version=eval_version)
        return result_str, result_dict

    def evaluation_map_segmentation(self, results):
        import torch

        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = 6 #len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]
            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None].cpu() >= thresholds
            label = label[:, :, None].cpu()

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluation_occ(self, logger, SC_metric, SSC_metric):

        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(SC_metric)
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(SSC_metric)
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)

        return eval_results

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts_tracking(self, batch_dict, pred_dicts, class_names, output_path=None):
        
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 
                'score': np.zeros(num_samples),
                'det_score': np.zeros(num_samples),
                'iou_score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 
                'pred_labels': np.zeros(num_samples),
                'tracking_id':np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            # pred_scores_det = box_dict['pred_scores_det'].cpu().numpy()
            # pred_scores_iou = box_dict['pred_scores_iou'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_ids = box_dict['track_ids'].cpu().numpy()
            pred_dict_tracking = get_template_prediction(pred_scores.shape[0])
            # pred_dict_det = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict_tracking

            pred_dict_tracking['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict_tracking['score'] = pred_scores
            # pred_dict_tracking['det_score'] = pred_scores_det
            # pred_dict_tracking['iou_score'] = pred_scores_iou
            pred_dict_tracking['boxes_lidar'] = pred_boxes
            pred_dict_tracking['pred_labels'] = pred_labels
            pred_dict_tracking['tracking_id'] = pred_ids

            return pred_dict_tracking

        annos = []
        # print('len pred_dict',len(pred_dicts))
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            else:
                print('ERROR',index, 'is empty')
            annos.append(single_pred_dict)
        try:
            assert len(annos) > 0
        except:
            print('CUDA ID',torch.cuda.current_device(), batch_dict['sample_idx'])
        return annos

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 
                'pred_labels': np.zeros(num_samples),
                'tracking_id':np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            # pred_ids = box_dict['track_ids'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            # pred_dict['tracking_id'] = pred_ids

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos
    
    def generate_prediction_dicts_with_tracking(self, batch_dict, pred_dicts, class_names, output_path=None):
        
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 
                'pred_labels': np.zeros(num_samples),
                'tracking_id':np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['eval_det_scores'].cpu().numpy()
            pred_boxes = box_dict['eval_det_boxes'].cpu().numpy()
            pred_labels = box_dict['eval_det_labels'].cpu().numpy()
            # pred_ids = box_dict['track_ids'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            # pred_dict['tracking_id'] = pred_ids

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam
        )

        # nuscenes_dataset = NuScenesDataset(
        #     dataset_cfg=dataset_cfg, class_names=None,
        #     root_path=ROOT_DIR / 'data' / 'nuscenes',
        #     logger=common_utils.create_logger(), training=True
        # )
        # nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
