import glob
import numpy as np
import torch
import torchvision
from PIL import Image
import random

from imagecorruptions import corrupt
from panopticapi.utils import rgb2id

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
        elif dataset == 'COCOval_corrupt':
            self.fnames = list(glob.glob(data_dir + '/*.jpg'))            
        else:
            raise NotImplementedError
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
        print(idx,self.total_idx)
        # st()
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
            panoptic_seg_id = rgb2id(np.array(panoptic_seg))
            # st()
        
        panoptic_seg_unique_id = np.unique(panoptic_seg_id)
        
        # st()
        # size_val = 32
        # st()
        mask_val = torch.zeros(self.num_protos,panoptic_seg_id.shape[0],panoptic_seg_id.shape[1])
        # st()

        for idx, unique_id in enumerate(panoptic_seg_unique_id):
            mask = (panoptic_seg_id == unique_id)
            # mask_resized = torchvision.transforms.Resize((size_val,size_val),torchvision.transforms.InterpolationMode.NEAREST)(torch.from_numpy(mask).unsqueeze(0))
            mask_resized = torch.from_numpy(mask)
            mask_val[idx] = mask_resized
        # st()
        self.total_idx += 1
        return self.transform(image, mask_val) 
