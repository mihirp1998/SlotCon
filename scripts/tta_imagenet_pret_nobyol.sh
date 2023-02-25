#!/bin/bash

set -e
set -x

data_dir='/projects/katefgroup/datasets/ObjectNet/objectnet-1.0/'
data_dir='/projects/katefgroup/datasets/imagenet_c/'
data_dir="/projects/katefgroup/datasets/ImageNet/"

output_dir="./output/tta_imagenet_val_10_1step_highler"
output_dir="./output/tta_imagenet_corrupt_motion_blur_5_lowlr_5step_2"
output_dir="./output/tta_imagenet_corrupt_snow_5_lowlr_5step_2"
output_dir="./output/tta_gauss_ce_full_joint_dtch_fix_weight13_track_smallparam"
# _tstats
# imagenet_corrupt-snow-5
# imagenetval
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
    --dataset imagenet_corrupt-gaussian_noise-5 \
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
    --optimizer sgd \
    --base-lr 2e-1 \
    --weight-decay 1e-5 \
    --warmup-epoch 0 \
    --epochs 100 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 0 \
    --seg-weight 0.0 \
    --tta-steps 20 \
    --do-tta \
    --no-load-optim \
    --class-weight 0.0 \
    --cont-weight 0.5  --do-only-classification --log-freq 5 --no-byol  --cross-entropy --joint-train  --detach-target --track-stats --custom-params 
    # --d
    # --custom-params --d
    # --d     --cls-joint-weight 1.0 \
    #  --d
    # --overfit 
    # --custom-params --track-stats --entropy-loss
    #  --d
    #  --d
    #   --d
    # --track-stats
    # --custom-params
    # --entropy-loss
    #  --d
    # --d
    #  --d
    # --d
# --resume output/imagenet_classify_73_pret_sup6_loaded/current.pth
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