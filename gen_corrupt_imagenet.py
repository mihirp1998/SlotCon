import glob
import os
import ipdb
st = ipdb.set_trace
import pickle
import numpy as np
from imagecorruptions import corrupt
from PIL import Image
folder_name = '/projects/katefgroup/datasets/ImageNet'

severity_level = 5
# st()
corruption_name = 'gaussian_noise'
save_folder_name = f"{folder_name}/val_{corruption_name}_{severity_level}"

filename = f'{folder_name}/val_list.p'
all_files = pickle.load(open(filename, 'rb'))

os.makedirs(save_folder_name, exist_ok=True)

for idx, file in enumerate(all_files):
    # st()
    folder_name = file.split('/')[-2]
    file_name = file.split('/')[-1]
    os.makedirs(f'{save_folder_name}/{folder_name}', exist_ok=True)
    print(f"Processing {idx} of {len(all_files)}")
    image = Image.open(file).convert('RGB')
    corrupt_image = corrupt(np.array(image), corruption_name=corruption_name, severity=severity_level)
    # st()
    corrupt_image = Image.fromarray(corrupt_image)
    corrupt_image.save(f'{save_folder_name}/{folder_name}/{file_name}')

fnames = glob.glob(f'{save_folder_name}/n*/*.JPEG')
pickle.dump(fnames,open(f'{save_folder_name}.p' ,'wb'))