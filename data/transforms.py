import math
import random
import warnings
import ipdb
import numpy as np
st = ipdb.set_trace
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageOps

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1+w1, j2+w2) - max(j1, j2))
    y_overlap = max(0, min(i1+h1, i2+h2) - max(i1, i2))
    return x_overlap * y_overlap

def _get_coord(i, j, h, w):
    coord = torch.Tensor([j, i, j + w, i + h])
    return coord

def _clip_coords(coords, params):
    x1_q, y1_q, x2_q, y2_q = coords[0]
    x1_k, y1_k, x2_k, y2_k = coords[1]
    _, _, height_q, width_q = params[0]
    _, _, height_k, width_k = params[1]

    x1_n, y1_n = torch.max(x1_q, x1_k), torch.max(y1_q, y1_k)
    x2_n, y2_n = torch.min(x2_q, x2_k), torch.min(y2_q, y2_k)

    coord_q_clipped = torch.Tensor([float(x1_n - x1_q) / width_q, float(y1_n - y1_q) / height_q,
                                    float(x2_n - x1_q) / width_q, float(y2_n - y1_q) / height_q])
    coord_k_clipped = torch.Tensor([float(x1_n - x1_k) / width_k, float(y1_n - y1_k) / height_k,
                                    float(x2_n - x1_k) / width_k, float(y2_n - y1_k) / height_k])
    return [coord_q_clipped, coord_k_clipped]


class GaussianBlur(nn.Module):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        super().__init__()
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(nn.Module):
    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)


class CustomTwoCrop(object):
    def __init__(self, size=224, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR,
                condition_overlap=True, mask_size=7):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.mask_size = mask_size

        self.scale = scale
        self.ratio = ratio
        self.condition_overlap = condition_overlap

    @staticmethod
    def get_params(img, scale, ratio, ):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            constraints list(tuple): list of params (i, j, h, w) that should be used to constrain the crop
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        for counter in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.01 * target_area: # at least 1 percent of the second crop is part of the first crop.
                    return i, j, h, w
        
        return self.get_params(img, scale, ratio) # Fallback to default option

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            crops (list of lists): result of multi-crop
        """
        crops, coords = [], []
        mask_crops = []
        # size_mask = (7,7)
        # size_mask = (224,224)        
        size_mask = (self.mask_size,self.mask_size)        
        # st()
        params1 = self.get_params(img, self.scale, self.ratio)
        coords.append(_get_coord(*params1))
        crops.append(TF.resized_crop(img, *params1, self.size, self.interpolation))
        # st()
        if mask is None:
            mask_crops.append(None)
        else:
            mask_crops.append(TF.resized_crop(mask, *params1, size_mask, TF.InterpolationMode.NEAREST))

        if not self.condition_overlap:
            params2 = self.get_params(img, self.scale, self.ratio)
        else:
            params2 = self.get_params_conditioned(img, self.scale, self.ratio, params1)
        
        coords.append(_get_coord(*params2))

        
        crops.append(TF.resized_crop(img, *params2, self.size, self.interpolation))
        # st()
        if mask is None:
            mask_crops.append(None)
        else:        
            mask_crops.append(TF.resized_crop(mask, *params2, size_mask, TF.InterpolationMode.NEAREST))

        return crops, _clip_coords(coords, [params1, params2]), mask_crops


class CustomRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, crops, coords, mask_crops):
        crops_flipped, coords_flipped, flags_flipped = [], [], []
        mask_crops_fipped = []
        for crop, coord, mask_crop in zip(crops, coords, mask_crops):
            crop_flipped = crop
            mask_crop_flipped = mask_crop
            coord_flipped = coord
            flag_flipped = False
            if torch.rand(1) < self.p:
                crop_flipped = TF.hflip(crop)
                if mask_crop is None:
                    mask_crop_flipped = None
                else:
                    mask_crop_flipped = TF.hflip(mask_crop)
                coord_flipped = coord.clone()
                coord_flipped[0] = 1. - coord[2]
                coord_flipped[2] = 1. - coord[0]
                flag_flipped = True

            crops_flipped.append(crop_flipped)
            mask_crops_fipped.append(mask_crop_flipped)
            coords_flipped.append(coord_flipped)
            flags_flipped.append(flag_flipped)

        return crops_flipped, coords_flipped, flags_flipped, mask_crops_fipped


class CustomDataAugmentation(object):
    def __init__(self, size=224, min_scale=0.08, mask_size=7,no_aug=False):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.no_aug = no_aug

        self.two_crop = CustomTwoCrop(size, (min_scale, 1), interpolation=TF.InterpolationMode.BICUBIC, mask_size=mask_size)
        if  self.no_aug:
            self.hflip = CustomRandomHorizontalFlip(p=0.0)
            self.global_transfo1 = transforms.Compose([
                normalize,
            ])            
            self.global_transfo2 = transforms.Compose([
                normalize,
            ])                        
        else:
            self.hflip = CustomRandomHorizontalFlip(p=0.5)

            # first global crop
            self.global_transfo1 = transforms.Compose([
                color_jitter,
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                color_jitter,
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                transforms.RandomApply([Solarize()], p=0.2),
                normalize,
            ])

        self.test_two_crop = CustomTwoCrop(size, (1, 1), interpolation=TF.InterpolationMode.BICUBIC, mask_size=mask_size)
        self.test_global_transfo = transforms.Compose([normalize])   

        self.normalize_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(size,size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(size=(mask_size,mask_size),interpolation=TF.InterpolationMode.NEAREST),
        ])                

    def __call__(self, image, mask):
        # st()
        # image_norm = self.normalize_image(image)
        # mask_norm = self.normalize_mask(mask)
        crops, coords, mask_crops = self.two_crop(image, mask)
        # st()
        
        crops, coords, flags, mask_crops = self.hflip(crops, coords, mask_crops)

        crops_transformed = []
        # st()
        # crops_transformed.append(self.global_transfo(crops[0]))
        # crops_transformed.append(self.global_transfo(crops[1]))        
        crops_transformed.append(self.global_transfo1(crops[0]))
        crops_transformed.append(self.global_transfo2(crops[1]))

        image_norm, _, mask_norm = self.test_two_crop(image, mask)
        image_norm = self.test_global_transfo(image_norm[0])
        mask_norm = mask_norm[0]

        # mask_crops = [mask_norm, mask_norm]
        # crops_transformed = [image_norm, image_norm]
        # st()


        # st()
        # image_norm = self.normalize_image(image)
        # mask_norm = self.normalize_mask(mask)
        # print(image_norm[0].shape)
        # print((np.array(image_norm[0]) == np.array(image_norm[1])).all())
        # st()
        return image_norm, mask_norm, crops_transformed, coords, flags, mask_crops




class TestTransform(object):
    def __init__(self,size,mask_size):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(size,size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(size=(mask_size,mask_size),interpolation=TF.InterpolationMode.NEAREST),
        ])        

        self.global_transfo = transforms.Compose([
            normalize,
        ])

    def __call__(self, image, mask):
        # st()
        image = self.global_transfo(image)
        mask = self.normalize_mask(mask)
        return image, mask