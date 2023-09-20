'''
Prepare ScanNet data for training.
author: Yinyu Nie
date: July, 2020
'''
import os
import math as m
import numpy as np
from process_utils import *
from tools import make_M_from_tqs
from tools import normalize
from tools import get_iou_cuboid
from tools import get_box_corners
import pickle
from plyfile import PlyData,PlyElement
from config import ShapeNet_Dir, CATID_scan2cad

SHAPENETCLASSES = ['void',
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']

# OBJ_CLASS_IDS = np.array([ 1,  7,  8, 13, 20, 31, 34, 43])
ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}


def write_ply(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


def generate(scan2cad_annotation, dataset_root):
    scene_name = scan2cad_annotation['id_scan']
    # print('Processing: %s.' % scene_name)
    scene_folder = os.path.join(dataset_root, scene_name)
    '''read orientation file'''
    meta_file = os.path.join(scene_folder, scene_name + '.txt')  # includes axis
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    Mscan = make_M_from_tqs(scan2cad_annotation["trs"]["translation"],
                            scan2cad_annotation["trs"]["rotation"],
                            scan2cad_annotation["trs"]["scale"])
    R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))
    # Mscan:               scan space --> world space
    # axis_align_matrix:   scan space --> origin space
    # R_transform:         world space --> origin space

    mesh_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scene_folder, scene_name + '.aggregation.json')
    seg_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.0.010000.segs.json')
    labelmap_file = 'LABELMAP.pkl'
    with open(labelmap_file, 'rb') as f:
        label_map = pickle.load(f)
    mesh_vertices, _, instance_bboxes, instance_bboxes, _, objid_to_ptid = \
        export(mesh_file, agg_file, seg_file, meta_file, label_map, None)
    '''preprocess boxes'''
    objid_pcd = {}
    for model in scan2cad_annotation['aligned_models']:
        # read corresponding shapenet scanned points
        catid_cad = model["catid_cad"]
        cls_id = SHAPENETCLASSES.index(ShapeNetIDMap[catid_cad[1:]])
        if cls_id != 1 or catid_cad != CATID_scan2cad:
            continue
        id_cad = model["id_cad"]
        obj_path = os.path.join(ShapeNet_Dir, catid_cad, id_cad + '/models/model_normalized.obj')
        assert os.path.exists(obj_path)
        obj_points = read_obj(obj_path)['v']
        
        '''transform shapenet obj to scannet'''
        t = model["trs"]["translation"]
        q = model["trs"]["rotation"]
        s = model["trs"]["scale"]
        Mcad = make_M_from_tqs(t, q, s)
        transform_shape = R_transform.dot(Mcad)
        # Mcad:             cad space --> world space
        # transform_shape:  cad space --> origin space
        '''get transformed axes'''
        center = (obj_points.max(0) + obj_points.min(0)) / 2.
        axis_points = np.array([center,
                                center - np.array([0, 0, 1]),
                                center - np.array([1, 0, 0]),
                                center + np.array([0, 1, 0])])

        axis_points_transformed = np.hstack([axis_points, np.ones((axis_points.shape[0], 1))]).dot(transform_shape.T)[
                                  ..., :3]
        center_transformed = axis_points_transformed[0]
        forward_transformed = axis_points_transformed[1] - axis_points_transformed[0]
        left_transformed = axis_points_transformed[2] - axis_points_transformed[0]
        up_transformed = axis_points_transformed[3] - axis_points_transformed[0]
        forward_transformed = normalize(forward_transformed)
        left_transformed = normalize(left_transformed)
        up_transformed = normalize(up_transformed)
        axis_transformed = np.array([forward_transformed, left_transformed, up_transformed])
        '''get rectified axis'''
        axis_rectified = np.zeros_like(axis_transformed)
        up_rectified_id = np.argmax(axis_transformed[:, 2])
        forward_rectified_id = 0 if up_rectified_id != 0 else (up_rectified_id + 1) % 3
        left_rectified_id = np.setdiff1d([0, 1, 2], [up_rectified_id, forward_rectified_id])[0]
        up_rectified = np.array([0, 0, 1])
        forward_rectified = axis_transformed[forward_rectified_id]
        forward_rectified = np.array([*forward_rectified[:2], 0.])
        forward_rectified = normalize(forward_rectified)
        left_rectified = np.cross(up_rectified, forward_rectified)
        axis_rectified[forward_rectified_id] = forward_rectified
        axis_rectified[left_rectified_id] = left_rectified
        axis_rectified[up_rectified_id] = up_rectified
        if np.linalg.det(axis_rectified) < 0:
            axis_rectified[left_rectified_id] *= -1
        '''deploy points'''
        obj_points = np.hstack([obj_points, np.ones((obj_points.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        coordinates = (obj_points - center_transformed).dot(axis_transformed.T)
        # obj_points = coordinates.dot(axis_rectified) + center_transformed
        '''define bounding boxes'''
        # [center, edge size, orientation]
        sizes = (coordinates.max(0) - coordinates.min(0))
        box3D = np.hstack([center_transformed, sizes[[forward_rectified_id, left_rectified_id, up_rectified_id]],
                             np.array([np.arctan2(forward_rectified[1], forward_rectified[0])])])
        # vectors = np.diag((coordinates.max(0) - coordinates.min(0)) / 2).dot(axis_rectified)
        # box3D = np.eye(4)
        # box3D[:3, :] = np.hstack([vectors.T, center_transformed[np.newaxis].T])

        '''to get instance id'''
        axis_rectified = np.array([[np.cos(box3D[6]), np.sin(box3D[6]), 0], [-np.sin(box3D[6]), np.cos(box3D[6]), 0], [0, 0, 1]])
        vectors = np.diag(box3D[3:6]/2.).dot(axis_rectified)
        scan2cad_corners = np.array(get_box_corners(box3D[:3], vectors))

        best_iou_score = 0.
        best_instance_id = 0 # means background points
        best_box = instance_bboxes[0]
        for inst_id, instance_bbox in enumerate(instance_bboxes):
            center = instance_bbox[:3]
            vectors = np.diag(instance_bbox[3:6]) / 2.
            scannet_corners = np.array(get_box_corners(center, vectors))
            iou_score = get_iou_cuboid(scan2cad_corners, scannet_corners)

            if iou_score > best_iou_score:
                best_iou_score = iou_score
                best_instance_id = inst_id + 1
                best_box = instance_bbox

        scannet_object_pts_id = np.array(objid_to_ptid[best_instance_id])
        scannet_object_pts = mesh_vertices[scannet_object_pts_id, 0:3]
        scannet_object_pts_transformed = np.hstack([scannet_object_pts, np.ones((scannet_object_pts.shape[0], 1))]).dot(np.linalg.inv(transform_shape).T)[..., :3]
        # scannet_object_pts_transformed[:, 2] = scannet_object_pts_transformed[:, 2] * -1
        # print(best_instance_id)
        # theta = 90
        # angle_matrix = np.matrix([[ m.cos(m.radians(theta)), -m.sin(m.radians(theta)), 0 ],
        #                           [ m.sin(m.radians(theta)), m.cos(m.radians(theta)) , 0 ],
        #                           [ 0           , 0            , 1 ]])
        # scannet_object_pts_transformed = scannet_object_pts_transformed.dot(angle_matrix.T)

        mu = best_box[:3]
        sigma = best_box[3:6]
        # scannet_object_pts_transformed = (scannet_object_pts_transformed-mu)/sigma

        # print(scannet_object_pts_transformed.shape)
        min_pt = scannet_object_pts_transformed.min(axis = 0)
        max_pt = scannet_object_pts_transformed.max(axis = 0)
        # min_pt = np.array(min_pt).squeeze(0)
        # max_pt = np.array(max_pt).squeeze(0)

        scale = max_pt - min_pt
        largest_index = np.argmax(scale)
        # scannet_object_pts_transformed[:, 0] = (scannet_object_pts_transformed[:, 0] - min_pt[0]) / (max_pt[0] - min_pt[0])
        # scannet_object_pts_transformed[:, 1] = (scannet_object_pts_transformed[:, 1] - min_pt[1]) / (max_pt[1] - min_pt[1])
        # scannet_object_pts_transformed[:, 2] = (scannet_object_pts_transformed[:, 2] - min_pt[2]) / (max_pt[2] - min_pt[2])
        scannet_object_pts_transformed[:, 0] = (scannet_object_pts_transformed[:, 0] - (max_pt[0] + min_pt[0]) / 2) / scale[largest_index]
        scannet_object_pts_transformed[:, 1] = (scannet_object_pts_transformed[:, 1] - (max_pt[1] + min_pt[1]) / 2) / scale[largest_index]
        scannet_object_pts_transformed[:, 2] = (scannet_object_pts_transformed[:, 2] - (max_pt[2] + min_pt[2]) / 2) / scale[largest_index]
        objid_pcd[best_instance_id-1] = scannet_object_pts_transformed
        
        # exit()
    return objid_pcd

