# Main process for extracting "desk" pointcloud from the ScanNet
# 1. Get the desk pointcloud and align & normalize it
# 2. Get several images of the corresponding item

# import statements
import os
import cv2
import open3d as o3d
import time
import shutil
import datetime
from SensorData import SensorData
from process_utils import *
from gen_object import *
import multiprocessing as mp
from main_process import manipulate_single_scene

# necessary data path
from config import Scene_Dir, ScanNet_Dir, ShapeNet_Dir, \
    Output_Dir, Cache_Dir, Log_File, CATNAME_scannet, \
    Save_colorimg, verbose


def mp_wrapper(scene: str):
    try:
        with open(Log_File, 'a') as log:
            log.write(f'{scene} starts at {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")} \n')
    except Exception as e:
        print(f'Writing logging of {scene} error: {e}!')
    manipulate_single_scene(scene)

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
    
    with mp.Pool(16) as p:
        p.map(mp_wrapper, scenes)

    end = time.time()

    info = f'[INFO] Finished at time {datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}, with time period {end-start} \n'  
    with open(Log_File, 'a') as log:
        log.write(info)
        print(info)