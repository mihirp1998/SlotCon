from panoptic_eval import COCOPanopticEvaluator
import json
import os
import torch
import glob
import ipdb
st = ipdb.set_trace
from PIL import Image
import numpy as np
from panopticapi.utils import get_traceback, rgb2id,id2rgb

panop_files = glob.glob('/projects/katefgroup/datasets/coco/annotations/semantic_val2017/*')

evaluator = COCOPanopticEvaluator('coco_2017_val_panoptic')
# gt_json = evaluator._metadata.panoptic_json
# with open(gt_json, "r") as f:
#     json_data = json.load(f)
evaluator.reset()
main_folder = '/projects/katefgroup/datasets/coco/'
root_folder = f'{main_folder}/annotations/'
gt_json_file = f'{root_folder}/panoptic_val2017.json'
gt_json_file = f'{root_folder}/semantic_val2017.json'

store_json_file = f'{root_folder}/mod_100_semantic_val2017.json'

with open(gt_json_file, 'r') as f:
    gt_json = json.load(f)

# st()
os.makedirs(f'{root_folder}/mod_100_semantic_val2017', exist_ok=True)
os.makedirs(f'{main_folder}/100_val2017', exist_ok=True)

mod_annotations = []

for index, annot in enumerate(gt_json['annotations'][:100]):
    filename = annot['file_name']
    panoptic_seg = np.array(Image.open(f'{root_folder}/semantic_val2017/{filename}').convert('RGB'))
    panoptic_seg_id = panoptic_seg[:,:,0]
    panoptic_seg_rgb = id2rgb(panoptic_seg_id)
    print(index)

    rgb_image = np.array(Image.open(f'{main_folder}/val2017/{filename}'.replace('png','jpg')).convert('RGB'))
    
    assert (panoptic_seg_id == rgb2id(panoptic_seg_rgb)).all()

    Image.fromarray(rgb_image).save(f'{main_folder}/100_val2017/{filename}'.replace('png','jpg'))
    Image.fromarray(panoptic_seg_rgb).save(f'{root_folder}/mod_100_semantic_val2017/{filename}')
    # assert (panoptic_seg[:,:,0] == panoptic_seg[:,:,1]).all()
    # assert (panoptic_seg[:,:,1] == panoptic_seg[:,:,2]).all()
    # for info_val in annot['segments_info']:
    #     info_val['id'] = info_val['category_id']
    mod_annotations.append(annot)

gt_json['annotations'] = mod_annotations

with open(store_json_file, 'w') as f:
    json.dump(gt_json, f)
# evaluator.evaluate()
# st()
# print('hello')