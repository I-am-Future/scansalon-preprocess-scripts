# Main process for extracting an category's pointcloud from the ScanNet
# 1. Get the desk pointcloud and align it
# 2. Export ball mesh / pointcloud corresponding to the object
# 3. Export several images of the corresponding item

# import statements
import os
import cv2
import time
import shutil
import datetime
from SensorData import SensorData
from process_utils import *
from gen_object import *
import open3d as o3d

from config import Scene_Dir, ScanNet_Dir, ShapeNet_Dir, \
    Output_Dir, Cache_Dir, Log_File, CATNAME_scannet, \
    Save_colorimg, Use_instanceimg_cache, verbose, verbose_critical, \
    Dump_ply, Dump_mesh_ball_radius, Dump_mesh_ball_resolu, \
    Dump_mesh_ball_color, Dump_num_pics

def manipulate_single_scene(scene: str):
    ''' Manipulate a single scene's pre-process. 
        @param scene <str>: The scene's folder name (also the scene id)
        @retval: status <int>: 0 means OK, failed otherwise
    '''

    ## 1. read-in aggregation and segmentation info
    objectid2segs, label2segs = read_aggregation(
            os.path.join(ScanNet_Dir, scene, f"{scene}_vh_clean.aggregation.json"))
    if verbose:
        print(label2segs)

    ## 2. get the camera's images 
    frame_skip = 20
    sd = SensorData(
            os.path.join(ScanNet_Dir, scene, f"{scene}.sens"), frame_skip)

    # export color images to cache, if needed in other places
    # by default, our program saves color_images directly, so we don't 
    # need to save it. 
    if Save_colorimg:
        if os.path.exists(f"{Cache_Dir}/colorimg/{scene}"):
            shutil.rmtree(f"{Cache_Dir}/colorimg/{scene}")
        sd.export_color_images(os.path.join(Cache_Dir, f'colorimg/{scene}/'))

    if verbose:
        print('finish loading color images')
    color_images = sd.get_color_images()
    num_frames = sd.num_frames

    ## 3. get the instance map
    instance_map = []
    if Use_instanceimg_cache and \
            os.path.exists(f"{Cache_Dir}/instanceimg/{scene}"):
        for i in range(0, num_frames, frame_skip):
            instance_map.append(cv2.imread(f"{Cache_Dir}/instanceimg/{scene}/{i}.png", 0))
    else:
        try:
            shutil.rmtree(f"{Cache_Dir}/instanceimg/{scene}")  
        except FileNotFoundError:
            pass
        os.makedirs(f"{Cache_Dir}/instanceimg/{scene}")
        os.system(f"unzip -q -j {ScanNet_Dir}/{scene}/{scene}_2d-instance-filt.zip -d {Cache_Dir}/instanceimg/{scene}")
        for i in range(0, num_frames, frame_skip):
            instance_map.append(cv2.imread(f"{Cache_Dir}/instanceimg/{scene}/{i}.png", 0))
    assert len(instance_map) == len(color_images)

    ## 4. get scan2cad data
    with open('scan2cad_download_link/full_annotations.json', 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        if data[i]['id_scan'] == scene:
            scan2cad = data[i]
            break
    else:
        print(f'No {scene} in scan2cad annotation dataset!')
        return 
        
    if verbose:
        print(scan2cad['id_scan'])

    ## 5. sort all desk segs
    objid_pcd = generate(scan2cad, ScanNet_Dir)
    if verbose:
        print('objid_pcd.keys()', objid_pcd.keys())

    for objid, (label, segs) in enumerate(label2segs):
        if verbose:
            print('objid:', objid, 'label:', label)
        if label not in CATNAME_scannet:
            continue 
        if objid not in objid_pcd.keys(): 
            continue

        pcd = objid_pcd[objid]

        if verbose_critical:
            print(f'Exporting object {scene}:{objid}-{label} with size {pcd.shape}', flush=True)

        if os.path.exists(os.path.join(Output_Dir, f"{scene}_{objid}")):
            shutil.rmtree(f"{Output_Dir}{scene}_{objid}")
        os.makedirs(os.path.join(Output_Dir, f"{scene}_{objid}"))

        # dump .ply file as well, if needed
        if Dump_ply:
            write_ply(f'{Output_Dir}{scene}_{objid}/{scene}_{objid}.ply', pcd)

        # add small ball at each point
        r = Dump_mesh_ball_radius
        spheres = o3d.geometry.TriangleMesh()
        for j in range(len(pcd)):
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=Dump_mesh_ball_radius, resolution=Dump_mesh_ball_resolu)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color(Dump_mesh_ball_color)
            T = np.eye(4)
            T[0:3, 3] = pcd[j]
            mesh_sphere.transform(T)
            spheres += mesh_sphere
        spheres.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh(f'{Output_Dir}{scene}_{objid}/{scene}_{objid}.obj', spheres, write_vertex_normals=False)

        # search img max occurance and dump ~5 jpg images
        # to assure quality, we select with maximum pixels
        num_pics = Dump_num_pics
        pixel_objid = np.zeros((len(instance_map), ))
        for i in range(len(instance_map)):
            pixel_objid[i] = np.sum(instance_map[i].flatten() == (objid+1))
        for i in find_K_largest(pixel_objid, num_pics):
            cv2.imwrite(f"{Output_Dir}{scene}_{objid}/{i}.jpg", color_images[i])
            mask = instance_map[i].copy()
            positive = (mask == (objid+1))
            negative = (mask != (objid+1))
            mask[positive] = 255
            mask[negative] = 0
            cv2.imwrite(f"{Output_Dir}{scene}_{objid}/{i}_mask.jpg", mask)

if __name__ == '__main__':
    if not os.path.exists(Output_Dir):
        os.makedirs(Output_Dir)
    if not os.path.exists(Cache_Dir):
        os.makedirs(Cache_Dir)

    # get the scenes directory name
    with open(Scene_Dir, 'r') as f:
        scenes = f.readlines()
    scenes = [scene.rstrip('\n') for scene in scenes]

    start = time.time()
    with open(Log_File, 'w') as log:
        info = f'[INFO] Starting... Total number of scenes is: {len(scenes)}; starting at time: {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} \n'
        log.write(info)
        print(info)
    
    for scene in scenes:
        with open(Log_File, 'a') as log:
            log.write(f'{scene} starts at {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} \n')
        manipulate_single_scene(scene)

    end = time.time()

    info = f'[INFO] Finished at time {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}, with time period {end-start} \n'  
    with open(Log_File, 'a') as log:
        log.write(info)
        print(info)
