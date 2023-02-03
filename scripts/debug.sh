#!/bin/bash

set -e
set -x
data_dir="/projects/katefgroup/datasets/msn_hard_pickled/train_set/"
data_dir="/scratch/coco"
output_dir="./output/coco_debug_quick"


CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12342 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset COCO \
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
    --batch-size 2 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 1 \
    --epochs 100 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 8 \
    --seg-weight 1.0 \
    --cont-weight 0.0 \
     --overfit --min-scale 1.0 --no-aug --d
    # --d
    # --overfit 
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2/current.pth
    # --overfit 
    # --d
    
