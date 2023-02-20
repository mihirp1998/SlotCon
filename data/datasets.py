import glob
import numpy as np
import torch
import torchvision
from PIL import Image
import random
import torch.nn.functional as F
from imagecorruptions import corrupt
from panopticapi.utils import rgb2id
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import yaml, json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import ipdb
st = ipdb.set_trace

from PIL import Image
from torch.utils.data import Dataset
import pickle
import copy
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data.build import get_detection_dataset_dicts
from panopticapi.utils import rgb2id
from detectron2.data import transforms as T


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []
    
    # st()

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation



class COCOPanopticNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """


    def __init__(
        self,
        transform_gen,
        is_train=True,
        args=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.img_format = 'RGB'
        self.is_train = is_train
        self.args = args
        # st()
        if is_train:
            self.tfm_gens = build_transform_gen(args.cfg, is_train)
        self.transform = transform_gen
        coco_dataset = 'coco_2017_train_panoptic'
        coco_dataset = 'coco_2017_val_panoptic'        
        self.coco_metadata = MetadataCatalog.get(coco_dataset)
        self.dataset_dicts = get_detection_dataset_dicts(
                (coco_dataset,),
                filter_empty=True,
                min_keypoints=0,
                proposal_files=None,
            )

    def __len__(self):
        return len(self.dataset_dicts)
    
    def __getitem__(self, idx):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        do_og_aug = False
        dataset_dict = self.dataset_dicts[idx]
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # st()
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if do_og_aug:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        image_shape = image.shape[:2]  # h, w
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = dataset_dict["segments_info"]

        if do_og_aug:
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        pan_seg_gt = rgb2id(pan_seg_gt)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])
        classes = np.array(classes)
        gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            gt_masks = masks.tensor
            gt_boxes = masks.get_bounding_boxes()

        # st()
        if not self.is_train:
            dataset_dict['instances'] = Instances(image_shape)
            dataset_dict['instances'].gt_masks = gt_masks
            dataset_dict['instances'].gt_boxes = gt_boxes
            dataset_dict['instances'].gt_classes = gt_classes
            dataset_dict["image"] = image
        else:
            image_pil = Image.fromarray(image.permute(1,2,0).numpy().astype(np.uint8))
            gt_masks = gt_masks.float()

            return_val = self.transform(image_pil, gt_masks) 

            image_norm, mask_norm,  crops, coords, flags, masks = return_val
            num_masks = mask_norm.shape[0]
            masks_norm_padded= F.pad(mask_norm, (0, 0, 0, 0, 0, 256 - num_masks), value=0)
            masks_padded= [F.pad(mask, (0, 0, 0, 0, 0, 256 - num_masks), value=0) for mask in masks]            
            all_instances = []
            for mask in masks:
                instances = Instances(image_shape)
                instances.gt_masks = mask
                # instances.gt_boxes = is.get_bounding_boxes()
                instances.gt_classes = gt_classes
                all_instances.append(instances)
            
            dataset_dict['instances'] = all_instances
            dataset_dict["image"] = crops
            dataset_dict["flags"] = flags
            dataset_dict["coords"] = coords
        dataset_dict["metadata"] = self.coco_metadata
        return dataset_dict



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
        st()
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
        
        if dataset == 'imagenetval':
            self.fnames = pickle.load(open('imagenet' + '/val_list.p','rb'))
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
            torch_mapping = json.load(open("torch_imgnet_cls_index.json",'r'))
            for key,val in torch_mapping.items():
                self.folder_index_mapping[val[0]] = [int(key)+1,val[1]]
        else:
            txtval = open('map_clsloc.txt','r')
            lines = txtval.readlines()
            dict_val = {}
            for line in lines:
                line = line.split(' ')
                dict_val[line[0]] = line[1:]            
            self.folder_index_mapping = dict_val
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
        class_label = torch.tensor(int(self.folder_index_mapping[foldername][0]) - 1 )
        class_str = self.folder_index_mapping[foldername][1]

        image = Image.open(fpath).convert('RGB')
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