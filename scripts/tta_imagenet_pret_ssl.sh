#!/bin/bash

set -e
set -x

data_dir="/projects/katefgroup/datasets/ImageNet/"
output_dir="./output/tta_imagenet_classify_73_pret_ssl"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12342 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset imagenetval_corrupt_gauss \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 2048 \
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
    --class-weight 0.0 \
    --tta-steps 5 \
    --do-tta \
    --no-load-optim \    
    --cont-weight 1.0  --do-only-classification --resume output/imagenet_classify_73_pret_ssl/current.pth --max-pool-classifier  --no-strict
    # --d 
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2_mask/current.pth --no-load-optim
    # --overfit --d --overfit --no-aug --min-scale 1.0