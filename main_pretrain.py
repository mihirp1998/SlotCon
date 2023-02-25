import argparse
import json
import os
import random
random.seed(0)

import shutil
import time
import datetime
import wandb

import ipdb
st = ipdb.set_trace
import logging as logger
import numpy as np
import torch
torch.manual_seed(0)

# import torch.distributed as dist
# import torch.backends.cudnn as cudnn

from data.datasets import ImageNet
from data.transforms import ClassificationPresetTrain,ClassificationPresetEval


from models import resnet
from models.slotcon import SlotCon
from utils.lars import LARS
# from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.util import AverageMeter

model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parser():
    parser = argparse.ArgumentParser('SlotCon')

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', help='dataset type')
    parser.add_argument('--test-dataset', type=str, default='COCOval100', help='dataset type')    
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--corrupt-name', type=str, default='', help='dataset director')
    parser.add_argument('--test-annot-dir', type=str, default='/projects/katefgroup/datasets/coco/annotations/mod_100_semantic_val2017', help='dataset director')
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
    
    parser.add_argument('--update-center-tta', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--update-teacher-tta', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    
    parser.add_argument('--center-loss', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')        
    parser.add_argument('--entropy-loss', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--track-stats', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars','adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=float, default=5., help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('--cont-weight', type=float, default=1.0, help='cont-weight')
    parser.add_argument('--cross-entropy-weight', type=float, default=0.2, help='cont-weight')    
    parser.add_argument('--seg-weight', type=float, default=0.0, help='seg weight')
    parser.add_argument('--cls-joint-weight', type=float, default=1.0, help='seg weight')
    parser.add_argument('--class-weight', type=float, default=0.0, help='classification weight')    
    parser.add_argument('--tta-steps', type=int, default=12, help='seg weight')    
    parser.add_argument('--do-only-classification', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--do-seg-class', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    
    parser.add_argument('--no-scheduler', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    
    parser.add_argument('--max-pool-classifier', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')        
    parser.add_argument('--no-strict', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')        
    parser.add_argument('--fine-tune', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')            
    parser.add_argument('--vit-probing', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')                
    parser.add_argument('--only-test', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    
    parser.add_argument('--do-5k', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')        
    parser.add_argument('--do-10', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')            
    parser.add_argument('--no-byol', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')                
    parser.add_argument('--merge-probs', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')                    
    parser.add_argument('--custom-params', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--wo-temp', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    
    parser.add_argument('--mse-loss', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')        
    parser.add_argument('--cross-entropy', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')            
    parser.add_argument('--mse-loss-inter', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--joint-train', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')    

    
    # misc
    parser.add_argument('--annot-dir', type=str, default='/projects/katefgroup/datasets/coco/annotations/mod_semantic_train2017/', help='output director')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')
    parser.add_argument('--log-freq', type=int, default=500, help='number of training epochs')    
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
    parser.add_argument('--detach-target', action='store_true', help='auto resume from current.pth')            

    args = parser.parse_args()
    # if os.environ["LOCAL_RANK"] is not None:
    #     args.local_rank = int(os.environ["LOCAL_RANK"])
    return args 

def get_optimizer(args, model):
    # st()
    if args.custom_params:
        trainable_params = []
        all_names = []
        for name, param in model.named_parameters():
            if "bn" in name:
                param.requires_grad = True
                trainable_params.append(param)
                all_names.append(name)
            else:
                param.requires_grad = False
    else:
        trainable_params = model.parameters()
    # st()
    if args.adam:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.base_lr,
            weight_decay=args.weight_decay)    
        # st()        
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=args.batch_size * args.world_size / 256 * args.base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'lars':

            optimizer = LARS(
                trainable_params,
                lr=args.batch_size * args.world_size / 256 * args.base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    return optimizer

def change_batchnorm_attr(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        if isinstance(model, torch.nn.SyncBatchNorm):
            # model.eval()
            model.track_running_stats = False
            model.running_mean = None
            model.running_var = None
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(change_batchnorm_attr(child))
            except TypeError:
                flatt_children.append(change_batchnorm_attr(child))
    return flatt_children


def build_model(args, do_test=False, skip_optmizer = False):
    encoder = resnet.__dict__[args.arch]
    model = SlotCon(encoder, args).cuda()


    if do_test:
        pass
    else:
        if args.track_stats:
            pass
        else:
            change_batchnorm_attr(model)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = get_optimizer(args,model)

    if skip_optmizer:
        return model
    else:
        return model, optimizer


def load_model_optim(args,model):

    resnet_saved = torch.load('resnet_saved_v21gpu.pth')
    # st()

    if args.track_stats:
        model.load_state_dict(resnet_saved)
    else:
        model.load_state_dict(resnet_saved,strict=False)
        change_batchnorm_attr(model)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = get_optimizer(args,model)
    return model,optimizer




def save_checkpoint(args, epoch, model, optimizer, scheduler, scaler=None):
    logger.info('==> Saving...')
    if scheduler is not None:
        state = {
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }
    else:
        state = {
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
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
    # resnet50_weights = torch.load('/home/mprabhud/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
    # st()
    # model.load_state_dict(checkpoint['model'])
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # st()

        model.re_init(args)

        if args.do_tta or args.fine_tune:
            for key_val in  checkpoint['model'].keys():
                if "_q" in key_val:
                    checkpoint['model'][key_val] = checkpoint['model'][key_val.replace('_q','_k')]
        # st()
        if args.no_strict:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint['model'])
        # st()
        if not args.no_load_optim:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1

        if args.fp16 and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        # st()
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))
        # st()

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
    elif args.arch == 'resnet50_pretrained_classification':
        mask_size = 7        
    else:
        raise NotImplementedError

    if args.joint_train:
        args.cls_joint_weight = 1.0 - args.cont_weight

    # prepare data
    import socket
    hostname = socket.gethostname()
    if 'imagenet' in args.dataset.lower():
        if 'grogu' in hostname:
            args.data_dir = '/grogu/datasets/imagenet'
    # st()
    transform = ClassificationPresetTrain(crop_size=176,auto_augment_policy='ta_wide')
    key_transform = ClassificationPresetTrain(crop_size=224,auto_augment_policy=None,random_erase_prob=0.)    
    eval_transform = ClassificationPresetEval(crop_size=224,resize_size=232)
    # transform = CustomDataAugmentation(args.image_size, args.min_scale, mask_size, args.no_aug)

    train_dataset = ImageNet(args.dataset, args.data_dir, transform, eval_transform, transform, overfit=args.overfit,do_tta=args.do_tta, batch_size=args.batch_size,tta_steps=args.tta_steps, args=args)
    test_dataset = ImageNet(args.test_dataset, args.data_dir, transform, eval_transform, transform, overfit=args.overfit,do_tta=args.do_tta, batch_size=args.batch_size,tta_steps=args.tta_steps, args=args)

    if args.joint_train:
        joint_dataset = ImageNet('ImageNet', args.data_dir, transform, eval_transform, transform, overfit=args.overfit,do_tta=args.do_tta, batch_size=args.batch_size,tta_steps=args.tta_steps, args=args)
        # joint_sampler = torch.utils.data.distributed.DistributedSampler(joint_dataset)
        joint_loader = torch.utils.data.DataLoader(
            joint_dataset, batch_size=args.batch_size, shuffle=(None is None), 
            num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)



    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(None is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)
    # st()
    # prepare test data
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(None is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model, optimizer = build_model(args)
    logger.info(model)
    # define scheduler
    if args.no_scheduler:
        scheduler = None
    else:
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
        model.global_steps = (args.start_epoch-1) * len(train_loader)

    
    if args.only_test:
        test(test_loader, model, args, scaler)
    else:
        for epoch in range(args.start_epoch, args.epochs + 1):
            # train_sampler.set_epoch(epoch)
            # train for one epoch
            train(train_loader, test_loader, joint_loader, model, optimizer, scaler, scheduler, epoch, args)

            if (epoch % args.save_freq == 0 or epoch == args.epochs):
                save_checkpoint(args, epoch, model, optimizer, scheduler, scaler)

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        if isinstance(model, torch.nn.SyncBatchNorm):
            model.track_running_stats = False
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def test(test_loader, model, args,scaler):
    model.eval()
    print('Testing')
    num_val = 1
    k1_acc = []
    q1_acc = []    
    model.eval()
    # st()
    # get_children(model)
    # for name, module in  model.named_modules():

    #     if 'bn' in name:
    #         # st()
    #         module.track_running_stats=False
    # st()
    with torch.no_grad():
        with torch.cuda.amp.autocast(scaler is not None):
            for i, batch in enumerate(test_loader):
                print(i,"example")
                image_norm, crops, class_labels, class_names, fpath = batch
                image_norm = image_norm.cuda(non_blocking=True)
                # mask_norm = mask_norm.cuda(non_blocking=True)    
                # st()            
                vis_dict = model((image_norm, class_labels, class_names), is_test=True)
                # st()
                # k1_acc.append(vis_dict['k1_classification_acc'])
                # q1_acc.append(vis_dict['q1_classification_acc'])
                # print(vis_dict['k1_classification_acc'],vis_dict['q1_classification_acc'])

                k1_acc = k1_acc + list(vis_dict['k1_classification_acc_unnorm'].cpu().numpy())  

                q1_acc = q1_acc + list(vis_dict['q1_classification_acc_unnorm'].cpu().numpy())  

                # st()
                if not args.only_test:
                    if i ==num_val:
                        break
                else:
                    vis_dict['k1_acc_avg'] = sum(k1_acc)/len(k1_acc)
                    vis_dict['q1_acc_avg'] = sum(q1_acc)/len(q1_acc)

                print("k1_acc_avg",vis_dict['k1_acc_avg'], "q1_acc_avg",vis_dict['q1_acc_avg'], list(vis_dict['k1_classification_acc_unnorm'].cpu().numpy()) )
                if not args.d:
                    wandb.log(vis_dict,step=model.global_steps)



def train(train_loader,test_loader, joint_loader, model, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # switch to train mode
    run_name = args.output_dir.split("/")[-1]
    model.train()
    model.train()
    num_params = count_parameters(model)
    print(f"num_params {num_params}")
    
    if args.override_lr:
        optimizer.param_groups[0]['lr'] = args.base_lr
    start_acc_q = []
    end_acc_q = []
    start_acc_k = []
    end_acc_k = []    
    end = time.time()

    # st()





    train_len = len(train_loader)
    for i, batch in enumerate(train_loader):
        image_norm,  crops, class_labels, class_str, fpath, imagenet_train, imagenet_clsidx = batch
        crops = [crop.cuda(non_blocking=True) for crop in crops]

        image_norm = image_norm.cuda(non_blocking=True)
        class_labels = class_labels.cuda(non_blocking=True)

        imagenet_train = imagenet_train.cuda(non_blocking=True)
        imagenet_clsidx = imagenet_clsidx.cuda(non_blocking=True)


        # st()
        
        if ((i%(args.tta_steps) == 0 or args.overfit) and args.do_tta):
            model_baseline,_ = build_model(args, True)
            # st()
            model_baseline.eval()
            
            with torch.no_grad():
                # with torch.cuda.amp.autocast(scaler is not None):
                # print("start")
                # model.re_init_test()
                vis_dict = model_baseline((image_norm, class_labels, class_str), is_test=True)

                class_acc_q, class_acc_k = vis_dict['q1_classification_acc'],vis_dict['k1_classification_acc']
                # st()
                start_acc_q.append(class_acc_q)
                start_acc_k.append(class_acc_k)
                # print("start",class_acc_q,class_acc_k)

                if not args.d:
                    wandb.log(vis_dict,step=model.global_steps)
            

            # refresh parameters
            model, optimizer  = load_model_optim(args,model)
            # model = build_model(args, skip_optmizer=True)
            num_params = count_parameters(model)
            print(f"num_params {num_params}")
        # st()


        model.train()
        model.train()        

        # print('training')

        loss, vis_dict = model((crops,class_labels,class_str, imagenet_train, imagenet_clsidx))
        # st()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        vis_dict['lr'] = optimizer.param_groups[0]['lr']

        if not args.d:
            wandb.log(vis_dict,step=model.global_steps)

        loss_meter.update(loss.item(), crops[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()


       

        if args.do_tta:
            if (i+1)%(args.tta_steps) == 0 and not args.overfit:
                model.eval()
                model.eval()
                with torch.no_grad():
                    # model.re_init_test()
                    # with torch.cuda.amp.autocast(scaler is not None):
                    vis_dict = model((image_norm, class_labels, class_str), is_test=True)
                    vis_dict['cont_loss'] = float("nan")


                    acc_q, acc_k = vis_dict['q1_classification_acc'],vis_dict['k1_classification_acc']


                    end_acc_q.append(acc_q)
                    end_acc_k.append(acc_k)
                    end_mean_q = torch.mean(torch.tensor(end_acc_q))
                    start_mean_q = torch.mean(torch.tensor(start_acc_q))

                    end_mean_k = torch.mean(torch.tensor(end_acc_k))
                    start_mean_k = torch.mean(torch.tensor(start_acc_k))

                    # model, optimizer = build_model(args)
                    # model.re_init()
                    # model.cuda()
                    # optimizer = get_optimizer(args,model)
                    print('reinitializing')                    

                    # model.set_means(start_mean_q, end_mean_q,start_mean_k, end_mean_k)
                    vis_dict['start_mean_q'] = start_mean_q
                    vis_dict['end_mean_q'] = end_mean_q
                    vis_dict['start_mean_k'] = start_mean_k
                    vis_dict['end_mean_k'] = end_mean_k


                    print(f"Mean Scores: Start- {start_mean_k},{start_mean_q}; End- {end_mean_k},{end_mean_q}")

                    if not args.d:
                        wandb.log(vis_dict,step=model.global_steps)
                        
                        

                        # load_checkpoint(args, model, optimizer, scheduler, scaler)
                  
                        print("end")

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - i)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{i}/{train_len}]  '
                f'Exp name: {run_name}  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}))  ')


if __name__ == '__main__':
    args = get_parser()

    run_name = args.output_dir.split("/")[-1]

    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # cudnn.benchmark = True

    args.world_size = 1
    args.batch_size = int(args.batch_size / args.world_size)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    # logger = setup_logger(output=args.output_dir,
    #                       distributed_rank=dist.get_rank(), name="slotcon")
    # if dist.get_rank() == 0:
    path = os.path.join(args.output_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    if not args.d:
        if run_name is not '':
            wandb.init(project='slot_con_6', entity="mihirp",id= run_name)
        else:
            wandb.init(project='slot_con_6', entity="mihirp")
        wandb.config.update(args)
    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )

    main(args)
    # switch 
