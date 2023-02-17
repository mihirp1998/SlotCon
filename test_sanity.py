import torch
from munch import Munch
import ipdb
from data.transforms import CustomDataAugmentation
from models import resnet
from models.slotcon import SlotCon
st = ipdb.set_trace
import torchvision.models as models

from data.datasets import ImageFolder, ImageNet

batch_size = 100
use_slotcon = True



args = Munch()
args.arch = 'resnet50_pretrained_classification'
args.dim_hidden =4096
args.dim_out =256
args.teacher_momentum = 0.99
args.sl_layer = False
args.do_only_classification = True
args.vit_probing = False
args.do_seg_class = False
args.group_loss_weight = 0.5
args.student_temp = 0.1
args.teacher_temp = 0.07
args.num_prototypes = 256
args.center_momentum = 0.9
args.world_size =1 
args.batch_size = batch_size
args.do_tta = False
args.start_epoch = 0
args.epochs = 100


transform = CustomDataAugmentation(224, 1.0, 224, True)

test_dataset = ImageNet('imagenet_corrupt-motion_blur-5', '/projects/katefgroup/datasets/imagenet_c/', transform,corrupt_name='',annot_dir='', overfit=False,do_tta=False, batch_size=batch_size,tta_steps=10,num_protos=256, args=args)
# st()
args.num_instances = len(test_dataset)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=0, pin_memory=True, sampler=None, drop_last=True)

if use_slotcon:
    encoder = resnet.__dict__[args.arch]
    resnet = SlotCon(encoder, args).cuda()
else:
    resnet = models.resnet50(pretrained=True)

# st()
resnet.cuda()
resnet.eval()
results = []
tmp_results = []
all_images = []
fpaths = []
class_label_list = []
for batch_idx, batch in enumerate(test_loader):
    print("batch_idx: ", batch_idx)
    image_norm, mask_norm,  crops, coords, flags, masks, class_labels, class_names, fpath = batch
    image_norm = image_norm.cuda()
    class_labels = class_labels.cuda()
    all_images.append(image_norm)
    fpaths.append(fpath)

    with torch.no_grad():
        if use_slotcon:
            output = resnet((image_norm,mask_norm,class_labels,class_names),is_test=True)
            # st()
            # results.append(output.argmax(-1))
            class_label_list.append(class_labels)
            tmp_results.append(output['k1_fc'].argmax(-1))
            results.append(output['k1_classification_acc_unnorm'])
        else:
            output = resnet(image_norm)
            print(output.shape)
            results.append(output.argmax(-1))            

# st()
class_label_list_new = torch.cat(class_label_list)
results_new = torch.cat(results)
tmp_results_new = torch.cat(tmp_results)
all_images_new = torch.cat(all_images)
# st()
print(class_label_list_new)
print(all_images_new.sum([1,2,3]))
print(results_new)
print(tmp_results_new)
print(fpaths)

# st()
print('hello')
#     st()
#     print('hello')
# print(batch_idx)