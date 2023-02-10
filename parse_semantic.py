from iopath.common.file_io import PathManager as PathManagerBase
from detectron2.data import detection_utils as utils
from panopticapi.utils import rgb2id
import numpy as np
from PIL import Image
import ipdb
import os
import cv2
st = ipdb.set_trace
import json
PathManager = PathManagerBase()
json_file = '/projects/katefgroup/datasets/coco/annotations/panoptic_train2017.json'
json_file = '/projects/katefgroup/datasets/coco/annotations/semantic_val2017.json'
with PathManager.open(json_file) as f:
    json_info = json.load(f)
image_dir = '/projects/katefgroup/datasets/coco/train2017'
gt_dir = '/projects/katefgroup/datasets/coco/annotations/semantic_val2017/'
import glob
all_files = glob.glob(gt_dir +"*.png")
for main_idx, file in enumerate(all_files):
    # image_id = int(ann["image_id"])
    # TODO: currently we assume image and label has the same filename but
    # different extension, and images have extension ".jpg" for COCO. Need
    # to make image extension a user-provided argument if we extend this
    # function to support other COCO-like datasets.
    # image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
    # label_file = os.path.join(gt_dir, ann["file_name"])
    # idx_mask = 1
    rgb_file = file.replace('semantic_','').replace('annotations','.').replace('png','jpg')
    rgb_image = np.array(Image.open(rgb_file).convert('RGB'))
    panoptic_seg = np.array(Image.open(file).convert('RGB'))
    assert (panoptic_seg[:,:,0] == panoptic_seg[:,:,1]).all()
    assert (panoptic_seg[:,:,1] == panoptic_seg[:,:,2]).all()
    all_idxs = np.unique(panoptic_seg[:,:,0])
    cv2.imwrite(f'dump/{main_idx:05d}_rgb.png',np.array(rgb_image))
    # st()
    for idx_val in all_idxs:
        mask_val = (panoptic_seg[:,:,0] == idx_val).astype(np.uint8)*255
        cv2.imwrite(f'dump/{main_idx:05d}_mask_{idx_val:05d}.png',mask_val)
    st()
    # panoptic_seg_id = rgb2id(np.array(panoptic_seg))    
    # print([i['id'] for i in ann['segments_info']])
    # pan_seg_gt = utils.read_image(label_file, "RGB")
    # st()
st()
print('hello')
# ipdb> np.unique(panoptic_seg)
# array([  0,   1,  62,  64,  67,  72,  78,  82,  84,  85,  86, 118, 119,
#        130, 156, 181, 186, 188, 189, 199, 200], dtype=uint8)