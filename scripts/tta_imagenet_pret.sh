#!/bin/bash

set -e
set -x

data_dir='/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/'
data_dir='/projects/katefgroup/datasets/imagenet_c/'
data_dir="/projects/katefgroup/datasets/ImageNet/"

output_dir="./output/tta_imagenet_val_10_1step_highler"
output_dir="./output/tta_imagenet_corrupt_motion_blur_5_lowlr_5step_2"
output_dir="./output/tta_imagenet_corrupt_snow_5_lowlr_5step_2"
output_dir="./output/tta_imagenet_corrupt_gaussian_noise_5_moresteps"
# imagenet_corrupt-snow-5
# imagenetval
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12341 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset imagenet_corrupt_mod-gaussian_noise-5 \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50_pretrained_classification \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 1000 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 128 \
    --optimizer sgd \
    --base-lr 0.01 \
    --weight-decay 0.0 \
    --warmup-epoch 0 \
    --epochs 100 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 0 \
    --seg-weight 0.0 \
    --tta-steps 50 \
    --do-tta \
    --no-load-optim \
    --class-weight 0.0 \
    --cont-weight 1.0  --do-only-classification --resume output/imagenet_classify_73_pret_sup6_loaded/current.pth
#  --do-10
# change optimizer to sgd
# change num_worker to 0
# add --do-tta
# add --tta-steps 5
# change cont-weight to 1.0
# change class-weight to 0.0
# add resume checkpoint_name
# add no-load-optim
# change base lr maybe
# change weight decay
# maybe play with the momentum factor of optimizer
# change warmup epoch to 0
# change dataset to imagenetval_corrupt_gauss
# remove test dataset
#  change cuda visible devices to 0 and nproc to 1
# change the output dir name