import glob
import os
import ipdb
st = ipdb.set_trace
import numpy as np
import sys
from imagecorruptions import corrupt
from PIL import Image
folder_name = '/projects/katefgroup/datasets/coco/val2017'
coco_files = glob.glob(f"{folder_name}/*.jpg")
severity_level = 5
# st()
# gaussian_noise, shot_noise, impulse_noise, defocus_blur,
#                     glass_blur, motion_blur, zoom_blur, snow, frost, fog,
#                     brightness, contrast, elastic_transform, pixelate,
#                     jpeg_compression, speckle_noise, gaussian_blur, spatter,
#                     saturate

# ----
#  val2017_fog_5 val2017_gaussian_noise_5  val2017_motion_blur_5/ val2017_snow_5/

corruption_name = sys.argv[1]
save_folder_name = f"{folder_name}_{corruption_name}_{severity_level}"

try:
    os.makedirs(save_folder_name)
except Exception as e:
    pass

for idx, file in enumerate(coco_files):
    print(f"Processing {idx} of {len(coco_files)}")
    image = Image.open(file).convert('RGB')
    corrupt_image = corrupt(np.array(image), corruption_name=corruption_name, severity=severity_level)
    # st()
    corrupt_image = Image.fromarray(corrupt_image)
    corrupt_image.save(file.replace(folder_name,save_folder_name))