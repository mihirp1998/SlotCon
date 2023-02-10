from panoptic_eval import COCOPanopticEvaluator
import json
import torch
import glob
import ipdb
st = ipdb.set_trace
from PIL import Image
import numpy as np
from panopticapi.utils import rgb2id
# panop_files = glob.glob('/projects/katefgroup/datasets/coco/annotations/semantic_val2017/*')
task_folder_name = 'panoptic_val2017'
task_folder_name = 'semantic_val2017'
task_folder_name = 'mod_100_semantic_val2017'
task_folder_name = 'mod_semantic_val2017'


root_folder = '/projects/katefgroup/datasets/coco/annotations/'
# json_file = f'{root_folder}/panoptic_val2017.json'
# json_file = f'{root_folder}/mod_100_semantic_val2017.json'
json_file = f'{root_folder}/{task_folder_name}.json'

evaluator = COCOPanopticEvaluator(f'{root_folder}/{task_folder_name}')
# gt_json = evaluator._metadata.panoptic_json
with open(json_file, "r") as f:
    json_data = json.load(f)

evaluator.reset()
print(len(json_data['annotations']))
image_ids = [annot['image_id'] for annot in json_data['annotations']]
for annot in json_data['annotations']:
    # st()
    filename = annot['file_name']
    panop_file = f'{root_folder}/{task_folder_name}/{filename}'
    panoptic_seg = rgb2id(np.array(Image.open(panop_file).convert('RGB')))
    # assert (panoptic_seg[:,:,0] == panoptic_seg[:,:,1]).all()
    # assert (panoptic_seg[:,:,1] == panoptic_seg[:,:,2]).all()
    # all_idxs = np.unique(panoptic_seg[:,:,0])
    outputs = {}
    inputs = {}
    inputs["file_name"] = panop_file.split('/')[-1]
    inputs["image_id"] = int(panop_file.split('/')[-1].split('.')[0])
    assert inputs["image_id"] == annot['image_id']  
    # st()
    outputs['panoptic_seg'] = (torch.from_numpy(panoptic_seg), None)
    
    evaluator.process([inputs], [outputs])
    # st()
    # print('hello')

# st()
results = evaluator.evaluate()
# st()
print('hello',results)