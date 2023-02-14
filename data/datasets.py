import glob
import numpy as np
import torch
import torchvision
from PIL import Image
import random

from imagecorruptions import corrupt
from panopticapi.utils import rgb2id
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import yaml, json


import ipdb
st = ipdb.set_trace

from PIL import Image
from torch.utils.data import Dataset
import pickle

class ImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform,
        annot_dir='',
        tta_steps=1,
        batch_size=1,
        num_protos=256,
        overfit=False,
        do_tta=False,

    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'kubrics':
            self.fnames = pickle.load(open(f'{data_dir}/files.p','rb'))
            self.data_dir = data_dir
            # st()
            # self.fnames = pickle.load()
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        elif dataset == 'COCOval100':
            # st()
            self.fnames = list(glob.glob(data_dir + '/100_val2017/*.jpg'))            
        elif dataset == 'COCOval_corrupt':
            self.fnames = list(glob.glob(data_dir + '/*.jpg'))            
        else:
            raise NotImplementedError
        self.dataset = dataset
        self.num_protos = num_protos
        # st()
        self.coco_cat_dict = {}
        for idx_val,cat in enumerate(COCO_CATEGORIES):
            self.coco_cat_dict[cat['id']] = cat
            self.coco_cat_dict[cat['id']]['main_index'] = idx_val

        self.batch_size = batch_size
        self.tta_steps = tta_steps
        self.annot_dir = annot_dir
        self.do_tta = do_tta
        self.overfit = overfit
        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform
        self.total_idx = 0

    def __len__(self):
        if self.do_tta:
            return 2000000
        else:
            return len(self.fnames)

    def __getitem__(self, idx):
        # print(idx,self.total_idx)
        if self.do_tta:
            num_iter = self.total_idx//self.batch_size 
            idx = num_iter//self.tta_steps
        # print(idx,self.total_idx)
        if  self.overfit:
            idx = 12
            idx = 4
              
        fpath = self.fnames[idx]
        # st()
        # fpath_seg = fpath.replace('coco/train2017', 'coco_mihir/annotations/semantic_train2017/').replace('.jpg', '.png')
        if self.dataset == 'kubrics':
            image_idx = fpath.split('/')[-1]
            fpath = f"{self.data_dir}/dataset_pickled/{image_idx}"
            pickled = pickle.load(open(fpath,"rb"))
            image = Image.fromarray(pickled['image'])
            panoptic_seg_id = np.squeeze(pickled['seg'],-1)
            # st()
        else:
            image = Image.open(fpath).convert('RGB')
            filename = fpath.split('/')[-1]
            panoptic_seg = Image.open(f'{self.annot_dir}/{filename}'.replace('jpg','png')).convert('RGB')
            # panoptic_seg_id = np.array(panoptic_seg)[:,:,0]
            panoptic_seg_id = rgb2id(np.array(panoptic_seg))
            # st()
        
        panoptic_seg_unique_id = np.unique(panoptic_seg_id)
        
        # st()
        # size_val = 32
        # st()
        mask_val = torch.zeros(self.num_protos,panoptic_seg_id.shape[0],panoptic_seg_id.shape[1])
        # st()
        class_labels = torch.ones(self.num_protos)*-1
        for idx, unique_id in enumerate(panoptic_seg_unique_id):
            mask = (panoptic_seg_id == unique_id)
            if unique_id == 0:
                class_labels[idx] = -1
            else:
                class_labels[idx] = unique_id 
            # mask_resized = torchvision.transforms.Resize((size_val,size_val),torchvision.transforms.InterpolationMode.NEAREST)(torch.from_numpy(mask).unsqueeze(0))
            mask_resized = torch.from_numpy(mask)
            mask_val[idx] = mask_resized
        # st()
        self.total_idx += 1
        return_val = self.transform(image, mask_val) 
        return_val = list(return_val)
        sup_labels = []
        # st()
        for class_label in class_labels:
            if class_label ==-1:
                sup_label = -1
            else:
                sup_label = self.coco_cat_dict[int(class_label)]['main_index']
            sup_labels.append(sup_label)
                
        # class_names = []
        # for class_label in class_labels:
        #     if class_label ==-1:
        #         class_name = ''
        #     else:
        #         class_name = self.coco_cat_dict[int(class_label)+1]['name']
        #     class_names.append(class_name)
        # st()
        return_val.append(torch.tensor(sup_labels))
        return_val.append(torch.tensor(class_labels))
        return_val.append(fpath.split('/')[-1])
        # st()
        return_val = tuple(return_val)

        return return_val

class ImageNet(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform,
        annot_dir='',
        tta_steps=1,
        batch_size=1,
        num_protos=256,
        corrupt_name='',
        overfit=False,
        do_tta=False,
        args = None

    ):
        super(ImageNet, self).__init__()
        # st()
        # if corrupt_name == '':
        #     self.corrupt_name = ''
        # else:
        #     self.corrupt_name, self.corrupt_level = corrupt_name.split('-') 
        self.do_banana = False
        
        
        if dataset == 'imagenetval':
            self.fnames = pickle.load(open('imagenet' + '/val_list.p','rb'))
        elif dataset =='imagenetbanana':
            self.do_banana = True
            self.fnames = glob.glob(f'/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/images/banana/*')
        elif dataset == 'imagenetval_corrupt_gauss':
            self.fnames = pickle.load(open('imagenet' + '/val_gaussian_noise_5.p','rb'))
        else:
            self.fnames = pickle.load(open('imagenet' + '/train_list.p','rb'))
        # with open(f'{data_dir}/train/index_synset.yaml', 'r') as file:
        #     indexfolder_mapping = yaml.safe_load(file)pickle.dump(self.fnames,open(data_dir + '/val_list.p','wb'))
        # self.folder_index_mapping = {v: k for k, v in indexfolder_mapping.items()}
        self.data_dir = args.data_dir
        # st()
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
            # st()
            # self.folder_index_mapping = pickle.load(open('map_clsloc.p','rb'))


        self.dataset = dataset
        self.num_protos = num_protos
        # st()
        self.batch_size = batch_size
        self.tta_steps = tta_steps
        self.annot_dir = annot_dir
        self.do_tta = do_tta
        self.overfit = overfit
        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform
        self.total_idx = 0

    def __len__(self):
        if self.do_tta:
            return 2000000
        else:
            return len(self.fnames)

    def __getitem__(self, idx):
        # print(idx,self.total_idx)
        if self.do_tta:
            num_iter = self.total_idx//self.batch_size 
            idx = num_iter//self.tta_steps
        # print(idx,self.total_idx)
        # st()

        if self.overfit:
            idx = 12
            idx = 4
            idx = 5
            idx = 6
        
        fpath = self.fnames[idx]
        main_filename = '/'.join(fpath.split('/')[-3:])
        fpath = self.data_dir + '/' + main_filename


        foldername = fpath.split("/")[-2] 
        if self.do_banana:
            class_label = torch.tensor(int(self.category_index_mapping['banana'][0]) - 1 )
            class_str = 'banana'
        else:
            class_label = torch.tensor(int(self.folder_index_mapping[foldername][0]) - 1 )
            class_str = self.folder_index_mapping[foldername][1]

        image = Image.open(fpath).convert('RGB')
        st()
        H,W = np.array(image).shape[:2]
    
        # if self.corrupt_name != '':
        #     image = corrupt(np.array(image), corruption_name=self.corrupt_name, severity=int(self.corrupt_level))

        mask_val = torch.zeros(1,H,W)
        return_val = self.transform(image, mask_val) 
        return_val = list(return_val)
        return_val.append(class_label)
        return_val.append(class_str)
        return_val.append(fpath.split('/')[-1])

        return_val = tuple(return_val)


        self.total_idx += 1
        return  return_val