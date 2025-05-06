from functools import partial
import numpy as np
import numba as nb
from skimage import transform
import torch
import torchvision
import os
import copy
from ...utils import box_utils, common_utils
try:
    from tools.visual_utils import open3d_vis_utils as V
except:
    pass


tv = None
try:
    import cumm.tensorview as tv
except:
    pass


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

# color_map = {
#     0: (0,0,0),
#     1: (255,255,255),
#     2: (255,0,0),
#     3: (0,255,0),
#     4: (0,0,255),
#     5: (255,255,0),
#     6: (0,255,255),
#     7: (255,0,255),
#     8: (192,192,192),
#     9: (128,128,128),
#     10: (128,0,0),
#     11: (128,128,0),
#     12: (0,128,0),
#     13: (128,0,128),
#     14: (0,128,128),
#     15: (0,0,128),
#     16: (128,128,128),
#     255:(0,0,0)
#  }

color_map = {  # RGB.
    # 0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}

name2cls = {  # RGB.
    # 0: (0, 0, 0),  # Black. noise
    'barrier': 1,  # Slategrey barrier
    'bicycle': 2,  # Crimson bicycle
    'bus': 3,  # Orangered bus
    'car': 4,  # Orange car
    'construction_vehicle': 5,  # Darksalmon construction
    'motorcycle': 6,  # Red motorcycle
    'pedestrian': 7,  # Blue pedestrian
    'traffic_cone': 8,  # Darkslategrey trafficcone
    'trailer': 9,  # Darkorange trailer
    'truck': 10,  # Tomato truck
}

colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [128, 128, 128, 255], # 16 for vis
], dtype=np.float32)

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            if points.shape[-1] == 6:
                seq_length = int(points[:,-1].max()) + 1
                voxels = []
                coordinates = []
                num_points = []
                for i in range(seq_length):
                    mask = points[:,-1] == i
                    mask_points = np.zeros_like(points[mask][:,:5])
                    mask_points = points[mask][:,:5]
                    voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(mask_points))
                    tv_voxels, tv_coordinates, tv_num_points = voxel_output
                    del mask_points
                    # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
                    # ...., -3,-2,-1,0
                    voxels.insert(0,tv_voxels.numpy())
                    coordinates.insert(0,tv_coordinates.numpy())
                    num_points.insert(0,tv_num_points.numpy())
            else:
                voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
                tv_voxels, tv_coordinates, tv_num_points = voxel_output
                # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
                voxels = tv_voxels.numpy()
                coordinates = tv_coordinates.numpy()
                num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        # import pdb;pdb.set_trace()
        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            if 'obj_points_shape' in data_dict:
                data_dict['obj_points'] = data_dict['points'][:data_dict['obj_points_shape']][mask[:data_dict['obj_points_shape']]]
                data_dict['obj_points_labels'] = data_dict['obj_points_labels'][mask[:data_dict['obj_points_shape']]]
                assert data_dict['obj_points'].shape[0] == data_dict['obj_points_labels'].shape[0],'path %s' % data_dict['frame_id']
            data_dict['points'] = data_dict['points'][mask]
            


        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )

            if 'num_sampled_gt_boxes' in data_dict:
                num_add_boxes = data_dict['num_sampled_gt_boxes']
                data_dict['sampled_gt_boxes'] = data_dict['gt_boxes'][-num_add_boxes:][mask[-num_add_boxes:]]
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            if 'gt_boxes_index' in data_dict.keys():
                data_dict['gt_boxes_index'] = data_dict['gt_boxes_index'][mask]
            # if 'gt_vis' in data_dict.keys():
            #     data_dict['gt_vis'] = data_dict['gt_vis'][mask]
            if 'gt_ids' in data_dict.keys():
                data_dict['gt_ids'] = data_dict['gt_ids'][mask]
            
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_normalize_sequence(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize_sequence, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        for i in range(len(data_dict['image_sequence'])):
            data_dict['image_sequence'][i]["camera_imgs"] = [compose(img) for img in data_dict['image_sequence'][i]["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict
    
    def image_calibrate_sequence(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate_sequence, config=config)
        
        
        for i in range(len(data_dict['image_sequence'])):
            img_process_infos = data_dict['image_sequence'][i]['img_process_infos']
            transforms = []
            for img_process_info in img_process_infos:
                resize, crop, flip, rotate = img_process_info

                rotation = torch.eye(2)
                translation = torch.zeros(2)
                # post-homography transformation
                rotation *= resize
                translation -= torch.Tensor(crop[:2])
                if flip:
                    A = torch.Tensor([[-1, 0], [0, 1]])
                    b = torch.Tensor([crop[2] - crop[0], 0])
                    rotation = A.matmul(rotation)
                    translation = A.matmul(translation) + b
                theta = rotate / 180 * np.pi
                A = torch.Tensor(
                    [
                        [np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)],
                    ]
                )
                b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
                b = A.matmul(-b) + b
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
                transform = torch.eye(4)
                transform[:2, :2] = rotation
                transform[:2, 3] = translation
                transforms.append(transform.numpy())
            data_dict['image_sequence'][i]["img_aug_matrix"] = transforms
        return data_dict
    
    def load_bev_segmentation(self, data_dict=None, config=None):
        if data_dict is None:
            from nuscenes.map_expansion.map_api import NuScenesMap
            xbound = config.xbound
            ybound = config.ybound
            patch_h = ybound[1] - ybound[0]
            patch_w = xbound[1] - xbound[0]
            canvas_h = int(patch_h / ybound[2])
            canvas_w = int(patch_w / xbound[2])
            config.patch_size = (patch_h, patch_w)
            config.canvas_size = (canvas_h, canvas_w)
            self.maps = {}
            for location in config.location:
                self.maps[location] = NuScenesMap(config.dataset_root, location)
            return partial(self.load_bev_segmentation, config=config)

        lidar_aug_matrix = data_dict["lidar_aug_matrix"].copy()
        # reverse flip 
        if 'flip_x' in data_dict and data_dict['flip_x']:
            lidar_aug_matrix[:3,:] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:]
        if 'flip_y' in data_dict and data_dict['flip_y']:
            lidar_aug_matrix[:3,:] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:]
        lidar2point = lidar_aug_matrix
        gt_masks_bev_list = []
        for i in range(data_dict["ref_from_car"].shape[0]):
            point2lidar = np.linalg.inv(lidar2point)
            lidar2ego = np.linalg.inv(data_dict["ref_from_car"][i])
            ego2global = np.linalg.inv(data_dict["car_from_global"][i])
            lidar2global = ego2global @ lidar2ego @ point2lidar

            map_pose = lidar2global[:2, 3]
            patch_box = (map_pose[0], map_pose[1], config.patch_size[0], config.patch_size[1])

            rotation = lidar2global[:3, :3]
            v = np.dot(rotation, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])
            patch_angle = yaw / np.pi * 180

            mappings = {}

            for name in config.classes:
                if name == "drivable_area*":
                    mappings[name] = ["road_segment", "lane"]
                elif name == "divider":
                    mappings[name] = ["road_divider", "lane_divider"]
                else:
                    mappings[name] = [name]

            layer_names = []
            for name in mappings:
                layer_names.extend(mappings[name])
            layer_names = list(set(layer_names))

            location = data_dict["location"]
            masks = self.maps[location].get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=layer_names,
                canvas_size=(config.canvas_size[0],config.canvas_size[1]),
            )
            masks = masks.astype(np.bool_)

            num_classes = len(config.classes)
            labels = np.zeros((num_classes, *config.canvas_size), dtype=np.int64)
            for k, name in enumerate(config.classes):
                for layer_name in mappings[name]:
                    index = layer_names.index(layer_name)
                    labels[k, masks[index]] = 1
            
            if 'flip_y' in data_dict and data_dict['flip_y']:
                labels = labels[:, :, ::-1].copy()
            if 'flip_x' in data_dict and data_dict['flip_x']:
                labels = labels[:, ::-1, :].copy()

            gt_masks_bev_list.append(labels)

        data_dict["gt_masks_bev"] = np.stack(gt_masks_bev_list)
        data_dict.pop('location')

        return data_dict

    def load_occupancy(self, data_dict=None, config=None):

        if data_dict is None:

            self.occ_path = config.dataset_root
            self.occ_grid_size = np.array(config.occ_grid_size)
            self.unoccupied = 0
            self.occ_pc_range = np.array(config.occ_pc_range)
            self.occ_voxel_size = (self.occ_pc_range[3:] - self.occ_pc_range[:3]) / self.occ_grid_size
            self.occ_gt_resize_ratio = 1
            self.use_vel = False
            return partial(self.load_occupancy, config=config)


        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(data_dict['scene_token'], data_dict['lidar_token'])
        #  [z y x cls] or [z y x vx vy vz cls]
        pcd = np.load(os.path.join(self.occ_path, rel_path))
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)  # x y z
        untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # rotate
        pcd_np_cor = (data_dict['lidar_aug_matrix'][:3,:3] @ pcd_np_cor.T).transpose(1,0)
        # shift
        pcd_np_cor = pcd_np_cor + data_dict['lidar_aug_matrix'][:3,3:4].T

        if 'obj_points' in data_dict:
            # rotate
            # obj_np_cor = (data_dict['lidar_aug_matrix'][:3,:3] @ data_dict['obj_points'][:,:3].T).transpose(1,0)
            # shift
            # obj_np_cor = obj_np_cor + data_dict['lidar_aug_matrix'][:3,3:4].T
            # obj_np_cor = self.world2voxel(obj_np_cor)
            obj_np_cor = data_dict['obj_points'][:,:3]

            # import pdb;pdb.set_trace()
            large_sampled_gt_boxes = box_utils.enlarge_box3d(
                data_dict['sampled_gt_boxes'][:, 0:7], extra_width=[0.0,0.0,0.0])
            pcd_np_cor,mask = box_utils.remove_points_in_boxes3d_occ(pcd_np_cor, large_sampled_gt_boxes)
            pcd_label = pcd_label[mask]
            pcd_np_cor = np.concatenate([obj_np_cor , pcd_np_cor], axis=0)
            points_label = np.array([name2cls[n] for n in data_dict['obj_points_labels']]).astype(np.int)
            pcd_label = np.concatenate([points_label[:,None], pcd_label], axis=0)
            assert pcd_label.shape[0] == pcd_np_cor.shape[0], 'path %s' % data_dict['index']
        # make sure the point is in the grid
        # suitfor flip
        # pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.occ_grid_size - 1)
        pcd_np_cor = self.world2voxel(pcd_np_cor)
        inside_scene_mask_x = np.logical_and(pcd_np_cor[:, 0] >= 0, pcd_np_cor[:, 0] <= self.occ_grid_size[0] - 1)
        inside_scene_mask_y = np.logical_and(pcd_np_cor[:, 1] >= 0, pcd_np_cor[:, 1] <= self.occ_grid_size[1] - 1)
        inside_scene_mask_z = np.logical_and(pcd_np_cor[:, 2] >= 0, pcd_np_cor[:, 2] <= self.occ_grid_size[2] - 1)
        inside_scene_mask = np.logical_and.reduce((inside_scene_mask_x, inside_scene_mask_y, inside_scene_mask_z))
        pcd_np_cor = pcd_np_cor[inside_scene_mask]
        pcd_label = pcd_label[inside_scene_mask]
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)
        # import pdb;pdb.set_trace()
        # pcd_color = np.array([color_map[int(l)]  for l in pcd_label if l !=0 and l != 255])/255
        # V.draw_scenes(pcd_np_cor,point_colors=pcd_color)
        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.occ_grid_size, dtype=np.uint8) * self.unoccupied
        processed_label = nb_process_label(processed_label, pcd_np)
        data_dict['gt_occ'] = processed_label
        data_dict['gt_occ_point'] = transformed_occ
        data_dict['gt_occ_label'] = pcd_label
        return data_dict

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.occ_voxel_size[None, :] + self.occ_pc_range[:3][None, :]

    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.occ_pc_range[:3][None, :]) / self.occ_voxel_size[None, :]

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
