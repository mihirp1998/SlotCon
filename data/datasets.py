import glob
import numpy as np
import torch
import torchvision

from panopticapi.utils import rgb2id
import ipdb
st = ipdb.set_trace
from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform
    ):
        super(ImageFolder, self).__init__()
        if dataset == 'ImageNet':
            self.fnames = list(glob.glob(data_dir + '/train/*/*.JPEG'))
        elif dataset == 'COCO':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg'))
        elif dataset == 'COCOplus':
            self.fnames = list(glob.glob(data_dir + '/train2017/*.jpg')) + list(glob.glob(data_dir + '/unlabeled2017/*.jpg'))
        elif dataset == 'COCOval':
            self.fnames = list(glob.glob(data_dir + '/val2017/*.jpg'))
        else:
            raise NotImplementedError
        # st()

        self.fnames = np.array(self.fnames) # to avoid memory leak
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        fpath_seg = fpath.replace('train2017', 'annotations/panoptic_train2017/').replace('.jpg', '.png')
        image = Image.open(fpath).convert('RGB')
        panoptic_seg = Image.open(fpath_seg).convert('RGB')
        panoptic_seg_id = rgb2id(np.array(panoptic_seg))
        panoptic_seg_unique_id = np.unique(panoptic_seg_id)
        
        # st()
        # size_val = 32
        # st()
        mask_val = torch.zeros(100,panoptic_seg_id.shape[0],panoptic_seg_id.shape[1])
        # st()

        for idx, unique_id in enumerate(panoptic_seg_unique_id):
            mask = (panoptic_seg_id == unique_id)
            # mask_resized = torchvision.transforms.Resize((size_val,size_val),torchvision.transforms.InterpolationMode.NEAREST)(torch.from_numpy(mask).unsqueeze(0))
            mask_resized = torch.from_numpy(mask)
            mask_val[idx] = mask_resized
        # st()
        return self.transform(image, mask_val) 
