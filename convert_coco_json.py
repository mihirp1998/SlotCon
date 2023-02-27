import json
import ipdb
st = ipdb.set_trace
gt_json = '/projects/katefgroup/datasets/coco/annotations/panoptic_train2017.json'
gt_json = '/home/mihir/phd_projects/slot_localloss/SlotCon/datasets/coco/annotations/panoptic_val2017.json'
with open(gt_json, "r") as f:
    json_data = json.load(f)
annotations_dict = {}
annotations = json_data['annotations'] 
for annot in annotations:
    annotations_dict[annot['image_id']] = annot

filename_final = gt_json.split('/')[-1]
gt_json_folder = '/'.join(gt_json.split('/')[:-1])

mod_filename = f'{gt_json_folder}/annotdict_{filename_final}'
# st()
json_data['annotations_dict'] = annotations_dict
# st()
print(mod_filename)
json.dump(json_data,open(mod_filename,'w'))
