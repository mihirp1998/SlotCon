import glob
import numpy as np
import torch
import torchvision
from PIL import Image
import random
random.seed(0)
torch.manual_seed(0)
import os

from imagecorruptions import corrupt
from panopticapi.utils import rgb2id
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import yaml, json


import ipdb
st = ipdb.set_trace

from PIL import Image
from torch.utils.data import Dataset
import pickle


class ImageNet(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform,
        eval_transform,
        key_transform,
        tta_steps=1,
        batch_size=1,
        overfit=False,
        do_tta=False,
        args = None

    ):
        super(ImageNet, self).__init__()
        self.do_objectnet = False
        self.corrupt_imgnet = False
        
        if dataset == 'imagenetval':
            self.fnames = pickle.load(open('imagenet' + '/val_list.p','rb'))
            if args.do_5k:
                if os.path.exists('imagenet' + '/val_list_5k.p'):
                    self.fnames = pickle.load(open('imagenet' + '/val_list_5k.p','rb'))
                else:
                    random.shuffle(self.fnames)
                    self.fnames = self.fnames[:5000]
                    pickle.dump(self.fnames, open('imagenet' + '/val_list_5k.p','wb'))
            if args.do_10:
                if os.path.exists('imagenet' + '/val_list_10.p'):
                    self.fnames = pickle.load(open('imagenet' + '/val_list_10.p','rb'))
                else:
                    random.shuffle(self.fnames)
                    self.fnames = self.fnames[:10]
                    pickle.dump(self.fnames, open('imagenet' + '/val_list_10.p','wb'))
            
        elif dataset =='imagenet_objectnet':
            self.do_objectnet = True
            self.objectnet_cats = json.load(open('mihir_objectnet_to_imagenet_1k.json','r'))
            if os.path.exists('objectnet_files_5k.p'):
                self.fnames = pickle.load(open('objectnet_files_5k.p','rb'))
            else:
                self.fnames = []
                for objectnet_cat in self.objectnet_cats:
                    objectdir = f'{args.data_dir}/images/{objectnet_cat}'
                    self.fnames += glob.glob(objectdir + '/*png')
                random.shuffle(self.fnames)
                pickle.dump(self.fnames[:5000],open('objectnet_files_5k.p','wb'))
            # self.fnames = glob.glob(f'/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/images/banana/*')
        elif 'imagenet_corrupt_mod' in dataset:            
            _, noise_type, noise_level = dataset.split('-')
            folder_name = f'{data_dir}/val_{noise_type}_{noise_level}/*/*.JPEG'
            filename = f'{data_dir}/{noise_type}_{noise_level}_1000.p'
            self.corrupt_imgnet = True
            if os.path.exists(filename):
                self.fnames = pickle.load(open(filename,'rb'))
            else:
                self.fnames = glob.glob(folder_name)
                random.shuffle(self.fnames)
                self.fnames = self.fnames[:1000]
                pickle.dump(self.fnames,open(filename,'wb'))
        elif 'imagenet_corrupt' in dataset:
            _, noise_type, noise_level = dataset.split('-')
            folder_name = f'{data_dir}/{noise_type}/{noise_level}/*/*.JPEG'
            filename = f'{data_dir}/{noise_type}_{noise_level}_1000.p'
            self.corrupt_imgnet = True
            if os.path.exists(filename):
                self.fnames = pickle.load(open(filename,'rb'))
            else:
                self.fnames = glob.glob(folder_name)
                random.shuffle(self.fnames)
                self.fnames = self.fnames[:1000]
                pickle.dump(self.fnames,open(filename,'wb'))
            # self.fnames = pickle.load(open('imagenet' + '/val_gaussian_noise_5.p','rb'))
        else:
            self.fnames = pickle.load(open('imagenet' + '/train_list.p','rb'))
        # with open(f'{data_dir}/train/index_synset.yaml', 'r') as file:
        #     indexfolder_mapping = yaml.safe_load(file)pickle.dump(self.fnames,open(data_dir + '/val_list.p','wb'))
        # self.folder_index_mapping = {v: k for k, v in indexfolder_mapping.items()}
        self.data_dir = data_dir
        if args.arch == 'resnet50_pretrained_classification':
            self.folder_index_mapping = {}
            self.category_index_mapping = {}
            torch_mapping = json.load(open("torch_imgnet_cls_index.json",'r'))
            for key,val in torch_mapping.items():
                self.folder_index_mapping[val[0]] = [int(key)+1,val[1]]
                self.category_index_mapping[val[1]] = [int(key)+1,val[0]]
        else:
            txtval = open('map_clsloc.txt','r')
            lines = txtval.readlines()
            dict_val = {}
            self.category_index_mapping = {}
            for line in lines:
                line = line.split(' ')
                dict_val[line[0]] = line[1:]
                self.category_index_mapping[line[2].replace('\n','')] = [int(line[1]),line[0]]
            self.folder_index_mapping = dict_val

        # self.folder_index_mapping = pickle.load(open('map_clsloc.p','rb'))
        self.dataset = dataset
        self.batch_size = batch_size
        self.tta_steps = tta_steps
        self.do_tta = do_tta
        self.overfit = overfit
        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.eval_transform = eval_transform
        self.transform = transform
        self.key_transform = key_transform
        self.total_idx = 0
        self.old_idx = 0

    def __len__(self):
        if self.do_tta:
            return len(self.fnames)*self.tta_steps*self.batch_size
        else:
            return len(self.fnames)

    def __getitem__(self, idx):
        if self.do_tta:
            num_iter = self.total_idx//self.batch_size 
            idx = num_iter//self.tta_steps
            if idx != self.old_idx:
                self.old_idx = idx
                print(f'Idx is: {self.old_idx}')


        if self.overfit:
            idx = 12
            idx = 4
            idx = 5
            idx = 6
        
        fpath = self.fnames[idx]
        if self.corrupt_imgnet:
            main_filename = '/'.join(fpath.split('/')[-4:])
        else:
            main_filename = '/'.join(fpath.split('/')[-3:])
        fpath = self.data_dir + '/' + main_filename


        foldername = fpath.split("/")[-2] 
        if self.do_objectnet:
            class_label = []
            class_str = []

            for object_val in self.objectnet_cats[foldername]:
                class_label.append(torch.tensor(int(self.folder_index_mapping[object_val[1]][0]) - 1 ))
                class_str.append(self.folder_index_mapping[object_val[1]][1])
            class_label = np.array(class_label)
                
        else:
            class_label = torch.tensor(int(self.folder_index_mapping[foldername][0]) - 1 )
            class_str = self.folder_index_mapping[foldername][1]

        image = Image.open(fpath).convert('RGB')
        
        if self.do_objectnet:
            image = Image.fromarray(np.array(image)[2:-2,2:-2])

            
        H,W = np.array(image).shape[:2]
        # st()
        return_val = []
        crops = []
        # st()
        image_eval = self.eval_transform(image)


        image_1 = self.transform(image) 
        crops.append(image_1)
        # st()
        image_2 = self.key_transform(image) 
        crops.append(image_2)
        # crops.append(image_eval)

        return_val.append(image_eval)
        return_val.append(crops)
        return_val.append(class_label)
        return_val.append(class_str)
        return_val.append(fpath.split('/')[-1])

        return_val = tuple(return_val)
        # st()

        self.total_idx += 1
        return  return_val