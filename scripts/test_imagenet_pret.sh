#!/bin/bash

set -e
set -x

data_dir='/projects/katefgroup/datasets/imagenet_c/'
data_dir="/projects/katefgroup/datasets/ImageNet/"
output_dir="./output/test_imagenet_classify_ours_motion_blur-5_fix_10"
output_dir="./output/tta_imagenet_corrupt_gaussian_noise_5"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12344 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset ImageNet \
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
    --batch-size 64 \
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
    --num-workers 8 \
    --seg-weight 0.0 \
    --class-weight 1.0 \
    --test-dataset imagenet_corrupt_mod-gaussian_noise-5 \
    --cont-weight 0.0  --do-only-classification --only-test --d --seed 0
    # --resume output/imagenet_classify_73_pret_sup6_loaded/current.pth 
    # --d
    # --do-10 --resume output/imagenet_classify_73_pret_sup6_loaded/current.pth
    # --d --d
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2_mask/current.pth --no-load-optim
    # --overfit --d --overfit --no-aug --min-scale 1.0