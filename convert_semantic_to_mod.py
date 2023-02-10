import glob
import os
import numpy as np
from PIL import Image
from panopticapi.utils import get_traceback, rgb2id,id2rgb
import ipdb
st = ipdb.set_trace
root_dir = '/projects/katefgroup/datasets/coco/annotations/'
task_folder_name = 'semantic_train2017'
out_folder_name = 'mod_' + task_folder_name
final_dir = f'{root_dir}/{out_folder_name}'
os.makedirs(final_dir, exist_ok=True)
print('glob start')
all_files = glob.glob(f'{root_dir}/{task_folder_name}/*')
print('glob end')
for idx, file in enumerate(all_files):
    rgb_image = np.array(Image.open(file).convert('RGB'))
    assert (rgb_image[:,:,0] == rgb_image[:,:,1]).all()
    assert (rgb_image[:,:,1] == rgb_image[:,:,2]).all()
    panoptic_seg_id = rgb_image[:,:,0]
    panoptic_seg_rgb = id2rgb(panoptic_seg_id)
    assert (panoptic_seg_id == rgb2id(panoptic_seg_rgb)).all()
    print(idx, len(all_files))
    Image.fromarray(panoptic_seg_rgb).save(file.replace(task_folder_name,out_folder_name))
    # st()