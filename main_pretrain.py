import argparse
import json
import os
import random
import shutil
import time
import datetime
import wandb

import ipdb
st = ipdb.set_trace

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from data.datasets import ImageFolder
from data.transforms import CustomDataAugmentation
from data.transforms import TestTransform

from models import resnet
from models.slotcon import SlotCon
from utils.lars import LARS
from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.util import AverageMeter

model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parser():
    parser = argparse.ArgumentParser('SlotCon')

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO', 'COCOplus', 'ImageNet','COCOval_corrupt','kubrics'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
   
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names, help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=float, default=5., help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('--cont-weight', type=float, default=1.0, help='cont-weight')
    parser.add_argument('--seg-weight', type=float, default=0.0, help='seg weight')
    parser.add_argument('--tta-steps', type=int, default=12, help='seg weight')    
    
    
    # misc
    parser.add_argument('--annot-dir', type=str, default='/scratch/coco_mihir/annotations/semantic_train2017/', help='output director')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')
    parser.add_argument('--log-freq', type=int, default=100, help='number of training epochs')    
    parser.add_argument('--d', action='store_true', default=False, help='do debug')    
    parser.add_argument('--overfit', action='store_true', default=False, help='do debug')        
    parser.add_argument('--sl-layer', action='store_true', default=False, help='second last layer')        
    parser.add_argument('--run-name', type=str, default='', help='run name')
    parser.add_argument('--do-tta', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--val-freq', type=int, default=1, help='save frequency')      
    parser.add_argument('--override-lr', action='store_true', help='auto resume from current.pth')      
    parser.add_argument('--adam', action='store_true', help='auto resume from current.pth')    
    parser.add_argument('--no-load-optim', action='store_true', help='auto resume from current.pth')    
    parser.add_argument('--no-aug', action='store_true', help='auto resume from current.pth')        
    parser.add_argument('--custom-lr', action='store_true', help='auto resume from current.pth')        

    args = parser.parse_args()
    if os.environ["LOCAL_RANK"] is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    return args 

def build_model(args):
    encoder = resnet.__dict__[args.arch]
    model = SlotCon(encoder, args).cuda()
    num_params = count_parameters(model)
    print(f"num_params {num_params}")
    # st()
    if args.do_tta:
        if args.adam:
            trainable_params = []
            for name, param in model.named_parameters():
                if "encoder_q" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            all_names = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    all_names.append(name)            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.base_lr,
                weight_decay=args.weight_decay)    
            # st()        
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=args.batch_size * args.world_size / 256 * args.base_lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
            elif args.optimizer == 'lars':
                # st()
                # trainable_params = []
                # for name, param in model.named_parameters():
                #     if "bn" in name and "encoder_q" in name:
                #         param.requires_grad = True
                #         trainable_params.append(param)
                #     else:
                #         param.requires_grad = False
                # all_names = []
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         all_names.append(name)

                # print("all_names", all_names)
                # st() model.parameters()

                optimizer = LARS(
                    model.parameters(),
                    lr=args.batch_size * args.world_size / 256 * args.base_lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
                # st()     
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.batch_size * args.world_size / 256 * args.base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'lars':
            optimizer = LARS(
                model.parameters(),
                lr=args.batch_size * args.world_size / 256 * args.base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    return model, optimizer


def save_checkpoint(args, epoch, model, optimizer, scheduler, scaler=None):
    logger.info('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.fp16:
        state['scaler'] = scaler.state_dict()
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    files = os.listdir(args.output_dir)    
    for f in files:
        if  (f'ckpt_epoch_{epoch}.pth' not in f) and (".pth" in f):
            os.remove(os.path.join(args.output_dir, f))

    shutil.copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))

def load_checkpoint(args, model, optimizer, scheduler, scaler=None):
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # st()
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        model.module.re_init(args)

        for key_val in  checkpoint['model'].keys():
            if "_q" in key_val:
                checkpoint['model'][key_val] = checkpoint['model'][key_val.replace('_q','_k')]

        model.load_state_dict(checkpoint['model'])
        # st()
        if not args.no_load_optim:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

        if args.fp16 and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        # st()
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))

        if args.override_lr:
            optimizer.param_groups[0]['lr'] = args.base_lr

    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.arch == 'resnet50':
        mask_size = 7
        mask_size = 28
    elif args.arch == 'resnet50_l2' or args.arch == 'resnet50_maskformer':
        mask_size = 56        
    elif args.arch == 'resnet50_l' or args.arch == 'resnet50_pretrained':
        mask_size = 28       
    else:
        raise NotImplementedError

    # prepare data
    transform = CustomDataAugmentation(args.image_size, args.min_scale, mask_size, args.no_aug)
    train_dataset = ImageFolder(args.dataset, args.data_dir, transform,annot_dir=args.annot_dir, overfit=args.overfit,do_tta=args.do_tta, batch_size=args.batch_size,tta_steps=args.tta_steps,num_protos=args.num_prototypes)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # prepare data
    # test_transform = TestTransform(args.image_size,mask_size)
    # test_dataset = ImageFolder(args.dataset, args.data_dir, test_transform, overfit=args.overfit,do_tta=args.do_tta)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=(test_sampler is None), 
    #     num_workers=0, pin_memory=True, sampler=test_sampler, drop_last=True)

    # st()

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model, optimizer = build_model(args)
    logger.info(model)
    # st()
    # define scheduler

    scheduler = get_scheduler(optimizer, len(train_loader), args)
    # define scaler
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')

    if args.resume:
        load_checkpoint(args, model, optimizer, scheduler, scaler)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, scheduler, epoch, args)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model, optimizer, scheduler, scaler)


def train(train_loader, model, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # switch to train mode

    model.train()
    num_params = count_parameters(model)
    print(f"num_params {num_params}")
    
    if args.override_lr:
        optimizer.param_groups[0]['lr'] = args.base_lr
    start_ari_q = []
    end_ari_q = []
    start_ari_k = []
    end_ari_k = []    
    end = time.time()

    # if args.do_tta:
    #     model.module._transfer_key_to_query()
    model.global_step = epoch * len(train_loader)

    # st()
    train_len = len(train_loader)
    for i, batch in enumerate(train_loader):
        # print(scheduler.after_scheduler.T_max)
        # print(f"num_iter {i}")
        image_norm, mask_norm,  crops, coords, flags, masks = batch
        # st()
        crops = [crop.cuda(non_blocking=True) for crop in crops]
        coords = [coord.cuda(non_blocking=True) for coord in coords]
        flags = [flag.cuda(non_blocking=True) for flag in flags]
        masks = [mask.cuda(non_blocking=True) for mask in masks]
        image_norm = image_norm.cuda(non_blocking=True)
        mask_norm = mask_norm.cuda(non_blocking=True)
        # st()

        if ((i%(args.tta_steps) == 0 or args.overfit) and args.do_tta):
            # st()
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(scaler is not None):
                    print("start")
                    # load_checkpoint(args, model, optimizer, scheduler, scaler)
                    # if (not args.overfit) and args.do_tta and i%(args.tta_steps) == 0 :
                    #     load_checkpoint(args, model, optimizer, scheduler, scaler)
                    # model.module._transfer_key_to_query()

                    (ari_q, ari_k),vis_dict = model((image_norm, mask_norm), is_test=True)
                    
                    
                    if args.do_tta:                
                        start_ari_q.append(ari_q)
                        start_ari_k.append(ari_k)

                    if not args.d and dist.get_rank() == 0:
                        wandb.log(vis_dict,step=model.module.global_steps)
                # st()
                # print("start")
                # break
        model.train()

        # compute output and loss
        with torch.cuda.amp.autocast(scaler is not None):
            loss, vis_dict = model((crops, coords, flags, masks))
        
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # print(optimizer.param_groups[0]['lr'])
        if not args.do_tta:
            scheduler.step()

        vis_dict['lr'] = optimizer.param_groups[0]['lr']

        if not args.d and dist.get_rank() == 0:
            wandb.log(vis_dict,step=model.module.global_steps)
        # avg loss from batch size
        loss_meter.update(loss.item(), crops[0].size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.do_tta:
            if (i+1)%(args.tta_steps) == 0 and not args.overfit:
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(scaler is not None):
                        # st()
                        (ari_q, ari_k),vis_dict = model((image_norm, mask_norm), is_test=True)
            
                        end_ari_q.append(ari_q)
                        end_ari_k.append(ari_k)
                        # st()
                        end_ari_mean_q = torch.mean(torch.tensor(end_ari_q))
                        start_ari_mean_q = torch.mean(torch.tensor(start_ari_q))

                        end_ari_mean_k = torch.mean(torch.tensor(end_ari_k))
                        start_ari_mean_k = torch.mean(torch.tensor(start_ari_k))
                        # st()
                        print(len(end_ari_q),len(start_ari_q),end_ari_q)
                        model.module.set_means(start_ari_mean_q, end_ari_mean_q,start_ari_mean_k, end_ari_mean_k)

                        # model.start_ari_mean = start_ari_mean
                        # model.end_ari_mean = end_ari_mean
                        # if not args.d  and dist.get_rank() == 0:
                        #     print(i,"loggin")
                        #     wandb.log({'end_ari_mean': end_ari_mean, 'start_ari_mean': start_ari_mean},step=i)

                        if not args.d and dist.get_rank() == 0:
                            wandb.log(vis_dict,step=model.module.global_steps)


                        load_checkpoint(args, model, optimizer, scheduler, scaler)

                        # st()                        
                        print("end")

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - i)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{i}/{train_len}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  ')


if __name__ == '__main__':
    args = get_parser()

    run_name = args.output_dir.split("/")[-1]

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    args.world_size = dist.get_world_size()
    args.batch_size = int(args.batch_size / args.world_size)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=args.output_dir,
                          distributed_rank=dist.get_rank(), name="slotcon")
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        if not args.d:
            if run_name is not '':
                wandb.init(project='slot_con_2', entity="mihirp",id= run_name)
            else:
                wandb.init(project='slot_con_2', entity="mihirp")
    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )

    main(args)
