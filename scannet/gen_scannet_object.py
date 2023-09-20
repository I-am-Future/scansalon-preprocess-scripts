'''
Prepare ScanNet data for training.
author: Yinyu Nie
date: July, 2020
'''
# import sys
# sys.path.append('.')
from configs.path_config import PathConfig, ShapeNetIDMap, SHAPENETCLASSES
import os
import pickle
from utils.scannet.load_scannet_object import export
import numpy as np
import math as m
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from utils.read_and_write import read_json, read_obj
from utils.scannet.tools import make_M_from_tqs, calc_Mbbox
from utils.shapenet import ShapeNetv2_path
from utils.scannet.tools import normalize
from utils.scannet.tools import get_iou_cuboid
from configs.path_config import ScanNet_OBJ_CLASS_IDS as OBJ_CLASS_IDS
from utils.scannet.tools import get_box_corners
from utils.pc_util import extract_pc_in_box3d

from plyfile import PlyData,PlyElement
def write_ply(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def get_votes(box3D, mesh_vertices, point_votes, indices, point_vote_idx):
    center = box3D[:3]
    orientation = box3D[6]
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    vectors = np.diag(box3D[3:6] / 2.).dot(axis_rectified)
    box3d_pts_3d = np.array(get_box_corners(center, vectors))
    # Find all points in this object's OBB
    pc_in_box3d, inds = extract_pc_in_box3d(mesh_vertices[..., :3], box3d_pts_3d)
    # Assign first dimension to indicate it is in an object box
    point_votes[inds, 0] = 1
    # Add the votes (all 0 if the point is not in any object's OBB)
    votes = np.expand_dims(center, 0) - pc_in_box3d[:, 0:3]
    sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
    for i in range(len(sparse_inds)):
        j = sparse_inds[i]
        point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
        # Populate votes with the fisrt vote
        if point_vote_idx[j] == 0:
            point_votes[j, 4:7] = votes[i, :]
            point_votes[j, 7:10] = votes[i, :]
    point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)

    return point_votes, point_vote_idx

def generate(scan2cad_annotation):
    scene_name = scan2cad_annotation['id_scan']
    print('Processing: %s.' % scene_name)
    output_dir = os.path.join(path_config.processed_data_path, scene_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file_bbox = os.path.join(output_dir, 'bbox.pkl')
    output_file_votes = os.path.join(output_dir, 'full_scan.npz')
    # if os.path.isfile(output_file_bbox) and os.path.isfile(output_file_votes):
    #     print('File already exists. skipping.')
    #     print('-' * 20 + 'done')
    #     return None

    '''initiate class averager'''
    mean_sizes = {}
    for cls_id in OBJ_CLASS_IDS:
        mean_sizes[cls_id] = []

    '''read orientation file'''
    meta_file = os.path.join(path_config.metadata_root, 'scans', scene_name, scene_name + '.txt')  # includes axis
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

    '''read scan file'''
    scene_folder = os.path.join(path_config.metadata_root, 'scans', scene_name)
    with open(path_config.raw_label_map_file, 'rb') as file:
        label_map = pickle.load(file)
    mesh_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scene_folder, scene_name + '.aggregation.json')
    seg_file = os.path.join(scene_folder, scene_name + '_vh_clean_2.0.010000.segs.json')
    mesh_vertices, _, instance_bboxes, instance_bboxes, _, objid_to_ptid = \
        export(mesh_file, agg_file, seg_file, meta_file, label_map, None)

    '''save mesh vertices with votes'''
    N = mesh_vertices.shape[0]
    point_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
    point_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
    indices = np.arange(N)

    '''preprocess boxes'''
    shapenet_instances = []
    object_count = 0
    for model in scan2cad_annotation['aligned_models']:
        # read corresponding shapenet scanned points
        catid_cad = model["catid_cad"]
        cls_id = SHAPENETCLASSES.index(ShapeNetIDMap[catid_cad[1:]])
        if cls_id not in path_config.OBJ_CLASS_IDS:
            continue
        id_cad = model["id_cad"]
        obj_path = os.path.join(ShapeNetv2_path, catid_cad, id_cad + '/models/model_normalized.obj')
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

        mean_sizes[cls_id].append(box3D[3:6])

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
        scannet_object_pts_transformed[:, 2] = scannet_object_pts_transformed[:, 2] * -1
        # theta = 90
        # angle_matrix = np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
        #                           [ m.sin(theta), m.cos(theta) , 0 ],
        #                           [ 0           , 0            , 1 ]])
        # scannet_object_pts_transformed = scannet_object_pts_transformed.dot(angle_matrix.T)

        mu = best_box[:3]
        sigma = best_box[3:6]
        scannet_object_pts_transformed = (scannet_object_pts_transformed-mu)/sigma

        class_name = SHAPENETCLASSES[cls_id]
        save_root = 'datasets/scannet_proc/{}'.format(class_name)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        write_ply('{}/{}_{}.ply'.format(save_root, scene_name, best_instance_id), scannet_object_pts_transformed)
        object_count += 1
        
        # shapenet_instances.append(
        #     {'box3D': box3D, 'cls_id': cls_id, 'shapenet_catid': catid_cad, 'shapenet_id': id_cad,
        #      'instance_id':best_instance_id, 'box_corners':scan2cad_corners})

        # '''to get point votes'''
        # point_votes, point_vote_idx = get_votes(box3D, mesh_vertices, point_votes, indices, point_vote_idx)

    if not len(shapenet_instances):
        return None
    # with open(output_file_bbox, 'wb') as file:
    #     pickle.dump(shapenet_instances, file, protocol=pickle.HIGHEST_PROTOCOL)

    # np.savez(output_file_votes, mesh_vertices=mesh_vertices, point_votes=point_votes, instance_labels=instance_labels)

    return mean_sizes

def batch_export():

    '''read scan2cad annotation file'''
    scan2cad_annotation_path = path_config.scan2cad_annotation_path
    scan2cad_annotations = read_json(scan2cad_annotation_path)

    p = Pool(processes=mp.cpu_count())
    mean_sizes_all = p.map(generate, scan2cad_annotations)
    p.close()
    p.join()

    return mean_sizes_all

if __name__ == '__main__':
    path_config = PathConfig('scannet')
    mean_sizes_all = batch_export()

    mean_size_array = np.zeros([len(OBJ_CLASS_IDS), 3])
    for idx, cls_id in enumerate(OBJ_CLASS_IDS):
        mean_size_array[idx] = np.mean(sum([item[cls_id] for item in mean_sizes_all if item is not None], []), axis=0)

    mean_size_output_file = os.path.join(path_config.metadata_root, 'scannet_means_with_instance_ids.npz')
    # np.savez(mean_size_output_file, arr_0=mean_size_array)

