#!/bin/bash

set -e
set -x

data_dir="/scratch/coco"
data_dir="/projects/katefgroup/datasets/ImageNet/"
output_dir="./output/tta_imagenet_classify_73_2_gauss_5_3"


CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12342 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset imagenetval_corrupt_gauss \
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
    --tta-steps 5 \
    --log-freq 1 \
    --do-only-classification \
    --do-tta --resume output/imagenet_classify_73_2/current.pth --no-load-optim --num-workers 0 --overfit
    # --overfit
    # --d
    #  --overfit 
    # --overfit --no-load-optim 
    # --adam --no-load-optim 
    #   --min-scale 0.5
    # --overfit 
    # --d