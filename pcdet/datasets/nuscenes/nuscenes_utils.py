"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import cv2
from copy import deepcopy
from prettytable import PrettyTable
from pcdet.utils import box_utils

nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
vehicle_classes = ['car', 'truck', 'construction_vehicle', 'bus','trailer', 'motorcycle', 'bicycle']

locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 canvas_size_road,
                 patch_size_road,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1,
                 class2label={
                     'road_divider': 0,
                     'lane_divider': 0,
                     'ped_crossing': 1,
                     'contours': 2,
                     'others': -1,
                 }):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.patch_size_road = patch_size_road
        self.canvas_size = canvas_size
        self.canvas_size_road = canvas_size_road
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation, patch_angle):
        map_pose = ego2global_translation[:2]
        # rotation = Quaternion(ego2global_rotation)
        # import pdb;pdb.set_trace()
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_box_road = (map_pose[0], map_pose[1],self.patch_size_road[0], self.patch_size_road[1])
        # patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(
            patch_box, patch_angle, self.line_classes, location)
        layer_names = ['drivable_area',]
        map_mask = self.nusc_maps[location].get_map_mask(patch_box_road, patch_angle, layer_names, self.canvas_size_road)
        if map_mask.shape[0]==1:
            pad_ones_sidewalk = np.zeros_like(map_mask)
            map_mask = np.concatenate([map_mask,pad_ones_sidewalk],0)

        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(
            patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        polygon_geom = self.get_map_geom(
            patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length,
                               self.class2label.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length,
                           self.class2label.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length,
                           self.class2label.get('contours', -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors,map_mask

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        # import pdb;pdb.set_trace()
        # pose_lists,line_lists,line_connector_lists  = self.map_explorer[location].map_api.discretize_centerlines(0.5)
        # geoms = self.map_explorer[location]._get_layer_line_center(patch_box, patch_angle, line_lists)
        # map_geom.append(('centerline', geoms))
        # geoms = self.map_explorer[location]._get_layer_line_center(patch_box, patch_angle, line_connector_lists )
        # map_geom.append(('line_connector', geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        line_vectors.append(self.sample_pts_from_line(l))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(
                poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(
                    line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(
                    line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(
                record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle,
                     patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle,
                     patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / \
                np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / \
                    np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(
        patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(
                coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(
                        new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, max_channel, thickness, angle_class):
    
    # import pdb;pdb.set_trace()
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(
        forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(
        backward_masks, filter_masks).sum(0).astype('int32')

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks

def preprocess_map_instance(vectors, patch_size, canvas_size, max_channel, thickness, angle_class):
    
    # import pdb;pdb.set_trace()
    confidence_levels = [-1]
    vector_num_list = {}
    # for i in range(max_channel):
    #     vector_num_list[i] = []
    k=0
    for _, vector in enumerate(vectors):
        if vector['pts_num'] >= 2:
            if k not in vector_num_list.keys():
                vector_num_list[k] = []
            vector_num_list[k].append(
                LineString(vector['pts'][:vector['pts_num']]))
            k += 1

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(k):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(
        forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(
        backward_masks, filter_masks).sum(0).astype('int32')

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks


def rasterize_map(vectors, patch_size, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel + 1):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                (LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels




def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
    

def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep

def get_map_mask(nusc_maps,patch_box,patch_angle,resolution,name):
    # patch_box = (300, 1700, 54, 54)
    # patch_angle = 0  # Default orientation where North is up
    layer_names = ['drivable_area', 'walkway']
    feat_size = int(patch_box[-1]/(0.075*resolution))
    canvas_size = (feat_size, feat_size)
    map_mask = nusc_maps[name].get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
    if map_mask.shape[0]==1:
        pad_ones_sidewalk = np.zero_like(map_mask)
        map_mask = np.concatenate([map_mask,pad_ones_sidewalk],0)
    # lanes = self.nusc_maps[name].get_records_in_radius(patch_box[0], patch_box[1], 54, layer_names)
    # lanes = lanes['drivable_area'] + lanes['walkway']
    # lanes = self.nusc_maps[name].discretize_lanes(lanes, discretization_meters=0.075)
    return map_mask


def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_cam=False):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.
    # import pdb;pdb.set_trace()
    nusc_maps = {}
    for map_name in locations:
        nusc_maps[map_name] = NuScenesMap(data_path, map_name=map_name)

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
    map_patch_size = (patch_h, patch_w)
    map_canvas_size = (canvas_h, canvas_w)
    road_canvas_size = (int(50*2/0.5), int(50*2/0.5))

    map_max_channel=3
    map_thickness=5
    map_angle_class=36

    vector_map = VectorizedLocalMap(
            dataroot=data_path,
            patch_size=map_patch_size,
            canvas_size=map_canvas_size,
            canvas_size_road=road_canvas_size,
            patch_size_road = (100,100)
        )

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()
        if sample['scene_token'] in train_scenes:
            continue
        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        )
        car_from_ref = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=False)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        )

        global_from_car = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=False,
        )

        # sample = nusc.get('sample', sample['token'])
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        global_from_ref = np.dot(global_from_car, car_from_ref)
        center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), global_from_ref.T)[0]
        angle = np.arctan2(global_from_ref[..., 1, 0], global_from_ref[..., 0,0])/np.pi *180
        center_global = np.dot(np.array([0,0,0,1]).reshape(1,4), global_from_ref.reshape((4, 4)).T)[0]
        angle = np.arctan2(global_from_ref.reshape((4, 4))[..., 1, 0], global_from_ref.reshape((4, 4))[..., 0,0])/np.pi *180
        # map_results = get_map_ann_info(map_name,center_global,angle)
        vectors, map_mask = vector_map.gen_vectorized_samples(map_name, center_global, angle)

        for vector in vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate(
                (pts, np.zeros((pts.shape[0], 1))), axis=1)

        for vector in vectors:
            vector['pts'] = vector['pts'][:, :2]

        semantic_masks, instance_masks,_,_ = preprocess_map(
            vectors, map_patch_size, map_canvas_size, map_max_channel, map_thickness, map_angle_class)

        num_cls = semantic_masks.shape[0]
        indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)
        assert semantic_indices.max() <= 3
        # ref is Lidar, car is ego
        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'scene_token': sample["scene_token"],
            'map_name':map_name,
            'sweeps': [],
            "ref_from_car": ref_from_car,
            "car_from_ref": car_from_ref,
            "car_from_global": car_from_global,
            "global_from_car": global_from_car,
            'ref_from_global': np.dot(ref_from_car, car_from_global),
            'global_from_ref': global_from_ref,
            'ego2global_translation': ref_pose_rec['translation'],
            'ego2global_rotation': ref_pose_rec['rotation'],
            'timestamp': ref_time,
            'road_mask': map_mask.astype('uint8'),
            'line_mask': semantic_indices.astype('uint8')

        }

        if ref_sd_rec['prev'] == "":
            info['is_first'] = True
        else:
            info['is_first'] = False

        if with_cam:
            info['cams'] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = ref_cs_rec["translation"],
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})
        

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)
            # import pdb;pdb.set_trace()
            instance_inds = np.array([nusc.getind('instance', ann['instance_token']) for ann in annotations])
            visibility_tokens = np.array([int(anno['visibility_token'])for anno in annotations])
            assert len(annotations) == len(gt_boxes) == len(velocity) == len(instance_inds)
            ############ for camera ########
            # info['lidar_valid_flag'] = mask
            # mask = np.ones_like(mask) > 0
            ############ for camera ########
            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask]
            info['instance_inds'] = instance_inds[mask]
            info['visibility_tokens'] = visibility_tokens[mask]
            bev_start_position = np.array([-49.75, -49.75,   0.])
            bev_resolution = np.array([0.5,  0.5, 20.])
            bev_dimension = np.array([200, 200,   1])
            instance_map = {}
            # instance_camera = np.zeros((bev_dimension[1], bev_dimension[0]))
            instance = np.zeros((bev_dimension[1], bev_dimension[0]))

            # import pdb;pdb.set_trace()
            vehicle_mask = np.isin(info['gt_names'], vehicle_classes)
            gt_bboxes_3d = info['gt_boxes'][vehicle_mask]
            # gt_labels_3d = info['gt_names'][vehicle_mask]
            instance_tokens = info['instance_inds'][vehicle_mask]

           ############ for camera ########
            # vis_tokens = info['visibility_tokens'][vehicle_mask]
            # visible_mask = (vis_tokens != 1)
            # gt_bboxes_3d = gt_bboxes_3d[visible_mask]
            # # gt_labels_3d = gt_labels_3d[visible_mask]
            # instance_tokens = instance_tokens[visible_mask]
            ############ for camera ########
            if len(gt_bboxes_3d) > 0:
                bbox_corners = box_utils.boxes_to_corners_3d(deepcopy(gt_bboxes_3d))[:, [0, 1, 2, 3], :2]
                bbox_corners = np.round((bbox_corners - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]).astype(np.int32)

                for index, instance_token in enumerate(instance_tokens):
                    if instance_token not in instance_map:
                        instance_map[instance_token] = len(
                            instance_map) + 1

                    # instance_id start from 1
                    instance_id = instance_map[instance_token]
                    poly_region = bbox_corners[index]
                    # cv2.fillPoly(segmentation, np.int32([poly_region]), 1.0)
                    cv2.fillPoly(instance, np.int32([poly_region]), instance_id)
                    # segmentations.append(segmentation)
                    # instances.append(instance)

            # import pdb;pdb.set_trace()
            # import matplotlib.pyplot as plt;plt.imshow(instance,cmap='jet');plt.show()
            info['instance_mask'] = instance.astype('uint8')


        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos

def transform_tracking_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    nusc_annos_det = {
        'results': {},
        'meta': None,
    }

    nusc_annos_iou = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        annos_det = []
        annos_iou = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )
        for k, box in enumerate(box_list):
            name = det['name'][k]
            if name in ['construction_vehicle','traffic_cone','barrier']:
                continue
            tracking_id = det['tracking_id'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr,
                "tracking_name": name,
                "tracking_score": box.score,
                "tracking_id": str(int(tracking_id)),
            }

            # nusc_anno_det = {
            #     'sample_token': det['metadata']['token'],
            #     'translation': box.center.tolist(),
            #     'size': box.wlh.tolist(),
            #     'rotation': box.orientation.elements.tolist(),
            #     'velocity': box.velocity[:2].tolist(),
            #     'detection_name': name,
            #     'detection_score': str(det['det_score'][k]),
            #     'attribute_name': attr,
            #     "tracking_name": name,
            #     "tracking_score": str(det['det_score'][k]),
            #     "tracking_id": str(int(tracking_id)),
            # }

            # nusc_anno_iou = {
            #     'sample_token': det['metadata']['token'],
            #     'translation': box.center.tolist(),
            #     'size': box.wlh.tolist(),
            #     'rotation': box.orientation.elements.tolist(),
            #     'velocity': box.velocity[:2].tolist(),
            #     'detection_name': name,
            #     'detection_score': str(det['iou_score'][k]),
            #     'attribute_name': attr,
            #     "tracking_name": name,
            #     "tracking_score": str(det['iou_score'][k]),
            #     "tracking_id": str(int(tracking_id)),
            # }
            annos.append(nusc_anno)
            # annos_det.append(nusc_anno_det)
            # annos_iou.append(nusc_anno_iou)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})
        nusc_annos_det['results'].update({det["metadata"]["token"]: annos_det})
        nusc_annos_iou['results'].update({det["metadata"]["token"]: annos_iou})

    return nusc_annos, nusc_annos_det, nusc_annos_iou


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details

def format_nuscene_results_tracking(metrics, class_names, version='default'):


    def metric_name_to_print_format(metric_name) -> str:
        """
        Get the standard print format (numerical precision) for each metric.
        :param metric_name: The lowercase metric name.
        :return: The print format.
        """
        if metric_name in ['amota', 'amotp', 'motar', 'recall', 'mota', 'motp']:
            print_format = '%.3f'
        elif metric_name in ['tid', 'lgd']:
            print_format = '%.2f'
        elif metric_name in ['faf']:
            print_format = '%.1f'
        else:
            print_format = '%d'
        return print_format

    result = '----------------Nuscene %s results-----------------\n' % version
    metric_names = metrics['label_metrics'].keys()
    result += 'Per-class results:\n'
    result += '\t\t'
    result += '\t'.join([m.upper() for m in metric_names]) + '\n'

    max_name_length = 7
    for class_name in class_names:
        print_class_name = class_name[:max_name_length].ljust(max_name_length + 1)
        result += '%s ' % print_class_name

        for metric_name in metric_names:
            val = metrics['label_metrics'][metric_name][class_name]
            print_format = '%f' if np.isnan(val) else metric_name_to_print_format(metric_name)
            result += '\t%s ' % (print_format % val)

        result += '\n'

    # Print high-level metrics.
    result +='\nAggregated results:\n'
    for metric_name in metric_names:
        val = metrics[metric_name]
        print_format = metric_name_to_print_format(metric_name)
        result += '%s\t%s\n' % (metric_name.upper(), print_format % val)
    result +='Eval time: %.1fs \n' % metrics['eval_time']
    result += '\n'
    details = {}
    details.update({
            "amota": metrics['amota'],
            "amotp": metrics['amotp'],
            "recall": metrics['recall'],
            "motar": metrics['motar'],
            "gt": metrics['gt'],
            "mota": metrics['mota'],
            "motp": metrics['motp'],
            "mt": metrics['mt'],
            "ml": metrics['ml'],
            "faf": metrics['faf'],
            "tp": metrics['tp'],
            "fp": metrics['fp'],
            "fn": metrics['fn'],
            "ids": metrics['ids'],
            "frag": metrics['frag'],
            "tid": metrics['tid'],
            "lgd": metrics['lgd']
            })
    return result, details

def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / union)
    
    return mean_ious

def format_SC_results(mean_ious, return_dic=False):
    class_map = {
        1: 'non-empty',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x


def format_SSC_results(mean_ious, return_dic=False):
    class_map = {
        0: 'free',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
