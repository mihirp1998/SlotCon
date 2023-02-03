#!/bin/bash

set -e
set -x

data_dir="/scratch/coco"
data_dir="/projects/katefgroup/datasets/coco_2017/val2017_gaussian_noise_5/"
output_dir="./output/tta_coco_joint_73_gaussian5_sgd_f_tta_step5_3_vis"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12342 --nproc_per_node=1 \
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
    --base-lr 0.01 \
    --weight-decay 0.0 \
    --warmup-epoch 0 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 1 \
    --save-freq 2 \
    --num-workers 8 \
    --seg-weight 0.0 \
    --cont-weight 1.0 \
    --annot-dir /projects/katefgroup/datasets/coco_2017/annotations/semantic_val2017/ \
    --tta-steps 5 \
    --log-freq 1 \
    --do-tta --resume output/coco_joint_73_new/current.pth --no-load-optim --num-workers 0
    # --overfit
    # --d
    #  --overfit 
    # --overfit --no-load-optim 
    # --adam --no-load-optim 
    #   --min-scale 0.5
    # --overfit 
    # --d