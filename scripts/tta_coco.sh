#!/bin/bash

set -e
set -x

data_dir="/scratch/coco"
data_dir="/projects/katefgroup/datasets/coco/val2017_gaussian_noise_5/"
data_dir="/projects/katefgroup/datasets/coco/val2017_snow_5/"
data_dir="/projects/katefgroup/datasets/coco/val2017_motion_blur_5/"
data_dir="/projects/katefgroup/datasets/coco/val2017_fog_5/"

output_dir="./output/tta_coco_gaussian_noise_5"
output_dir="./output/tta_coco_snow_5"
output_dir="./output/tta_coco_motion_blur_5"
output_dir="./output/tta_coco_fog_5"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12339 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset COCOval_corrupt \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50_pretrained \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 64 \
    --optimizer sgd \
    --base-lr 1.0 \
    --weight-decay 0.0 \
    --warmup-epoch 0 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --num-workers 0 \
    --seg-weight 0.0 \
    --cont-weight 0.0 \
    --annot-dir /projects/katefgroup/datasets/coco/annotations/semantic_val2017/ \
    --tta-steps 100 \
    --do-tta --resume output/coco_joint_73_new/current.pth --no-load-optim --d
    # --overfit
    # --d
    #  --overfit 
    # --overfit --no-load-optim 
    # --adam --no-load-optim 
    #   --min-scale 0.5
    # --overfit 
    # --d