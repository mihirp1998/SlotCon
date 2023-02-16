import json
import ipdb
st = ipdb.set_trace
import glob
from collections import defaultdict
import pickle
import json


# with open('imagenet_tmp.txt') as f:
# # with open('../mappings/imagenet_to_label_2012_v2.txt') as f:    
#     labels = [line.strip() for line in f.readlines()]    
import json
objectnet_to_imagenet = json.load(open('/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/mappings/objectnet_to_imagenet_1k.json','r'))
imagenet_txt = open('imagenet1000_clsidx_to_labels.txt','r')
imagenet_lines = imagenet_txt.readlines()
all_imagenet_categories = []
for line_val in imagenet_lines:
    line_val_strip = line_val[:-1]
    line_val_strips = line_val_strip.split(', ')
    all_imagenet_categories.append(line_val_strips)
    # st()
objectnet_mapping = defaultdict(lambda: [])


folder_index_mapping = {}
torch_mapping = json.load(open("torch_imgnet_cls_index.json",'r'))
for key,val in torch_mapping.items():
    folder_index_mapping[key] = [key,val[0],val[1]]


folder_object_label_mapping = json.load(open("/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/mappings/folder_to_objectnet_label.json",'r'))
object_label_folder_mapping = {}
for key,val in folder_object_label_mapping.items():
    object_label_folder_mapping[val] = key



for objectnet_key, objectnet_imagenet_classes in objectnet_to_imagenet.items():
    objectnet_imagenet_classes_list = objectnet_imagenet_classes.split('; ')
    for class_val in objectnet_imagenet_classes_list:
        for index_imagenet, imagenet_classes in enumerate(all_imagenet_categories):
            if class_val in imagenet_classes:
                # st()
                objectnet_mapping[object_label_folder_mapping[objectnet_key]].append(folder_index_mapping[str(index_imagenet)])
                # st()
                break
                # st()
# st()
json.dump(objectnet_mapping,open('mihir_objectnet_to_imagenet_1k.json','w'))
# pickle.dump(objectnet_mapping,open('objectnet_to_imagenet_1k.p','wb'))

# folder_index_mapping = {}
# torch_mapping = json.load(open("torch_imgnet_cls_index.json",'r'))
# for key,val in torch_mapping.items():
#     folder_index_mapping[val[0]] = [int(key)+1,val[1]]

# pytorch_to_objectnet = json.load(open('imagenet_pytorch_id_to_objectnet_id.json','r'))
# all_categories = glob.glob('/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/images/*')
# category_names = [x.split('/')[-1] for x in all_categories]
# category_names.sort()

# for key_imagenet_pytorch, val_objectnet in pytorch_to_objectnet.items():
#     print(category_names[val_objectnet],torch_mapping[key_imagenet_pytorch][1])
#     # st()




# st()
print('hello')