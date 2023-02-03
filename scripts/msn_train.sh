#!/bin/bash

set -e
set -x
data_dir="/scratch/coco"
data_dir="/projects/katefgroup/datasets/msn_hard_pickled/train_set/"
output_dir="./output/msn_train_cont_unload4"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12342 --nproc_per_node=4 \
    main_pretrain.py \
    --dataset kubrics \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50_pretrained \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 37 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 32 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 0.5 \
    --epochs 80 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --num-workers 4 \
    --seg-weight 0.0 \
    --cont-weight 1.0  --resume output/msn_train2//ckpt_epoch_2.pth
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2/current.pth
    # --overfit 
    # --d
    
