'''
Configs for pre-processing
'''

## config for cache
Use_instanceimg_cache = True  # when set to True, use cache exported last time. Otherwise re-export
Save_colorimg = False         # save color images, not necessary in our case

## config for dump options
Dump_ply = True
Dump_mesh_ball_radius = 0.005
Dump_mesh_ball_resolu = 3
Dump_mesh_ball_color = [0.6, 0.6, 0.6]
Dump_num_pics = 5

## config for directories
Scene_Dir = './ALL_SCENES.txt'   # under current directory
ScanNet_Dir = '/path/to/ScanNet/scans/'
ShapeNet_Dir = '/path/to/ShapeNetCore.v2/'
Output_Dir = '/path/to/output/scannet_desk/'
Cache_Dir = '/path/to/cache/'

## config for log
Log_File = 'LOGGING.txt'  # log file
verbose = False           # stdout debug info 
verbose_critical = True   # stdout critical info

## working-on objects

work_category = 'desk'

if work_category == 'desk':
    # scan2cad category id. If the category id is not this, skip it directly
    # so we can speed up the process. (Used in gen_object.py).
    CATID_scan2cad = '04379243'

    # scannet categoty name. If the segment's category name in this, do it 
    # so we are dealing with desk. (Used in main_process.py).
    CATNAME_scannet = ['desk', 'table', 'nightstand']

    ## shapenet category id. If the category id is not this, skip it directly
    CLSID_shapenet = 1

elif work_category == 'chair':

    CATID_scan2cad = '03001627'
    CATNAME_scannet = ['chair', 'office chair', 'armchair', 'sofa chair', 'stack of chairs', 'folded chair', 'beanbag chair', 'folded chairs', 'massage chair', 'recliner chair', 'rocking chair', 'stack of folded chairs', 'stool', 'couch']
    CLSID_shapenet = 7

elif work_category == 'sofa':

    CATID_scan2cad = '04256520'
    CATNAME_scannet = ['couch', 'sofa']
    CLSID_shapenet = 13

elif work_category == 'lamp':

    CATID_scan2cad = '03636649'
    CATNAME_scannet = ['lamp', 'lamp base', 'desk lamp', 'wall lamp', 'table lamp', 'ceiling lamp', 'night lamp']
    CLSID_shapenet = 19

elif work_category == 'bed':

    CATID_scan2cad = '02818832'
    CATNAME_scannet = ['bed', 'mattress', 'loft bed', 'sofa bed', 'bunk bed', 'air mattress']
    CLSID_shapenet = 32

else:
    raise NotImplementedError
