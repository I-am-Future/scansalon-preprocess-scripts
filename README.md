# README

The [ScanSalon](https://github.com/yushuang-wu/SCoDA) Dataset pre-process scripts (currently only for 5 categories from ScanNet).

The object's point clouds from ScanNet are extracted and normalized by this scripts, and then, they are sent to artists to build corresponding meshes.

## Install

There are several trivial packages needed, such as `numpy`, `open3d`, `quaternion`, `plyfile`, `cv2`, `imageio`. All of them can be installed by `pip/conda`.

## File Structure

+ `LABELMAP.pkl`, `ALL_SCENES.txt`, `scan2cad_download_link`, `scannet` (from [here](https://github.com/GAP-LAB-CUHK-SZ/RfDNet/tree/main/utils/scannet)) are metadata.
+ `config.py` stores configurations for exporting point clouds and meshes.
+ `main_process.py` is the main scripts, with `main_process_mp.py` adds multiprocessing support. 
+ Other remaining files are utility tools in exporting. 

## Basic Program Workflow

Majorly in `gen_object.py`.

1. Read the ScanNet scene point clouds, metadata.
2. Read the ShapeNet scene meshes.
3. Match shapenet meshes and ScanNet point clouds in the "origin space"
4. After matching, we have the right transformation to get ScanNet point clouds to the "CAD space", which is aligned with axis.
5. Finally, we normalize it so that maximum-spanned axis are in range `[-0.5, 0.5]`.
6. Export some auxiliary images and instance map as well, which helps artists to create the model.

You can refer to this script to implement code that can convert our ScanSalon data back to the original ScanNet coordinate. 

## Running

Set up necessary path in the `config.py`:

+ `Scene_Dir`: keep as default.
+ `ScanNet_Dir`: path where the `scans` folder in `ScanNet` locates. It looks like:

```text
scans
|-scene0000_00
| |-scene0000_00.txt
| |-scene0000_00_2d-instance-filt.zip
| |-scene0000_00_vh_clean_2.ply
| |-scene0000_00_vh_clean.aggregation.json
| `-......
|-scene0000_01
`-......
```

+ `ShapeNet_Dir`: path where ShapeNet root locates.

```
ShapeNetCore.v2
|-02747177
| |-10839d0dc35c94fcf4fb4dee5181bee
| | `-models......
| `-......
|-02808440  
|-02871439
`-......
```

+ `Output_Dir` a path where the exported data locates.
+ `Cache_Dir`: some data, such as instance map, can be saved in the `Cache_Dir`. So next time, you can directly use it.

Use `main_process.py` to run the pre-processing in one thread (so it may be quite slow). You can use `main_process_mp.py` to utilize multiprocessing support. 