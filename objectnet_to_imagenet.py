import json
import ipdb
st = ipdb.set_trace
import glob



with open('imagenet_tmp.txt') as f:
# with open('../mappings/imagenet_to_label_2012_v2.txt') as f:    
    labels = [line.strip() for line in f.readlines()]    
st()

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




st()
print('hello')