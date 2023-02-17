#!/bin/bash

set -e
set -x

data_dir="/projects/katefgroup/datasets/ImageNet/"
data_dir='/projects/katefgroup/datasets/imagenet_c/'
output_dir="./output/test_imagenet_classify_ours_motion_blur-5_fix_10"

CUDA_VISIBLE_DEVICES=0 python main_pretrain_eval.py \
    --dataset imagenet_corrupt-motion_blur-5 \
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
    --batch-size 50 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 3 \
    --epochs 100 \
    --fp16 \
    \
    --print-freq 40 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 0 \
    --seg-weight 0.0 \
    --class-weight 1.0 \
    --test-dataset imagenet_corrupt-motion_blur-5 \
    --cont-weight 0.0  --do-only-classification --only-test --d
    # --d
    # --do-10 --resume output/imagenet_classify_73_pret_sup6_loaded/current.pth
    # --d --d
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2_mask/current.pth --no-load-optim
    # --overfit --d --overfit --no-aug --min-scale 1.0