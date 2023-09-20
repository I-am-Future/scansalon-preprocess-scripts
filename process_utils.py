# Utilities of main process for extracting "desk" pointcloud from the ScanNet

# import statements
import os
import re
import json
import csv
import numpy as np
import quaternion
from plyfile import PlyData, PlyElement


def read_mesh_vertices(filename):
    ''' read XYZ for each vertex. '''
    assert os.path.isfile(filename), f"[ERROR] {filename} is NOT a file. "
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    ''' read XYZ RGB for each vertex.
        Note: RGB values are in 0-255
    '''
    assert os.path.isfile(filename), f"[ERROR] {filename} is NOT a file. "
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def read_aggregation(filename):
    ''' read aggregation json data, '''
    assert os.path.isfile(filename), f"[ERROR] {filename} is NOT a file. "
    object_id_to_segs = {}
    label_to_segs = []
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            label_to_segs.append((label, segs))
    return object_id_to_segs, label_to_segs

def read_aggregation1(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename), f"[ERROR] {filename} is NOT a file. "
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def find_K_largest(arr: np.ndarray, k: int):
    ''' '''
    arr = arr.copy()
    maxargs = []
    for i in range(k):
        if np.max(arr) == 0:
            break
        maxarg = np.argmax(arr)
        maxargs.append(maxarg)
        arr[maxarg] = -np.inf
    return maxargs

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M

def SE3_inv(M):
    ans = np.zeros((4,4))
    ans[:3, :3] = M[:3, :3].T
    ans[:3, 3] = - (M[:3, :3].T).dot(M[:3, 3])
    ans[3, 3] = 1/M[3, 3]
    return ans

def transform_pcd(scene_annotation: dict):
    transforms = []
    for model in scene_annotation['aligned_models']:
        if model["catid_cad"] == '04379243':  # it is a desk
            print(model["id_cad"])
            '''transform shapenet obj to scannet'''
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]
            Mcad = make_M_from_tqs(t, q, s)
            # print(Mcad)
            transforms.append(Mcad)
    return transforms

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r', encoding="utf-8")
    data = {}
    for head in flags:
        data[head] = []
    for line in fid:
        line = line.strip()
        if not line:
            continue
        line = re.split('\s+', line)
        if line[0] in flags:
            data[line[0]].append(line[1:])
    fid.close()
    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float32)
    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float32)
    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float32)
    return data

def read_json(file):
    '''
    read json file
    :param file: file path.
    :return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output

def export(mesh_file, agg_file, seg_file, meta_file, label_map, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation1(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]

    objid_to_ptid = {}
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                         xmax - xmin, ymax - ymin, zmax - zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox
        objid_to_ptid[obj_id] = (instance_ids == obj_id).tolist()

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        np.save(output_file + '_sem_label.npy', label_ids)
        np.save(output_file + '_ins_label.npy', instance_ids)
        np.save(output_file + '_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids, \
           instance_bboxes, object_id_to_label_id, objid_to_ptid

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices




# test
if __name__ == '__main__':
    # print(read_aggregation('scene0000_00/scene0000_00_vh_clean.aggregation.json'))
    # print(read_segmentation('scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json'))
    with open('scan2cad_download_link/full_annotations.json', 'r') as f:
        data = json.load(f)

    scene = data[1441]
    # for i in range(len(data)):
    #     if data[i]['id_scan'] == 'scene0000_00':
    #         print(i)
    print(scene['id_scan'])
    transform_pcd(scene)
    print(make_M_from_tqs(scene['trs']['translation'], scene['trs']['rotation'], scene['trs']['scale']))