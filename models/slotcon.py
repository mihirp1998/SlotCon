import math
import torch
import ari
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import matplotlib.pyplot as plt
import wandb
import numpy as np
from detectron2.modeling import build_model
from timm.models.vision_transformer import PatchEmbed, Block



from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from hungarian_matcher import HungarianMatcher
from detectron2.evaluation import COCOPanopticEvaluatorExampleBased
import copy
import ipdb
st = ipdb.set_trace




class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


def summ_instance_masks(masks, image, pred=False):
    # if self.save_this or overrirde:
    # st()
    image_cpu = image.cpu()
    masks_cpu = masks.cpu()
    masks_cpu = F.interpolate(masks_cpu.float().unsqueeze(0),image_cpu.shape[1:],mode='nearest').squeeze(0)
    masks_cpu = masks_cpu.squeeze(1)
    # st()
    if pred:
        # masks_cpu = torch.sigmoid(masks_cpu).round()
        old_shape = masks_cpu.shape
        num_slots = masks_cpu.shape[0]
        masks_cpu = torch.argmax(masks_cpu.reshape(masks_cpu.shape[0],-1).transpose(1,0),axis=-1)
        # st()
        masks_cpu = F.one_hot(masks_cpu,num_slots).float().transpose(1,0).reshape(old_shape)

    num_slots_c = torch.sum(masks_cpu.sum([1,2])>0.0)

    farthest_colors = plt.get_cmap("rainbow")([np.linspace(0, 1, num_slots_c)])[:,:,:3][0]
    rgb_canvas = torch.ones([3,masks_cpu.shape[-2],masks_cpu.shape[-1]])
    # aggregated_rgb_canvas = torch.zeros([3,masks_cpu.shape[-2],masks_cpu.shape[-1]])
    start_idx = 0
    for index, mask in enumerate(masks_cpu):
        # if masks_bool[index] ==1.:
        if torch.sum(mask) > 0:
            chosen_color = farthest_colors[start_idx].reshape([3,1])
            start_idx += 1
            # print(chosen_color)
            indicies = torch.where(mask == 1.0)
            rgb_canvas[:,indicies[0],indicies[1]] = torch.from_numpy(chosen_color).float()
    # st()
    rgb_canvas = rgb_canvas 
    rgb_canvas = rgb_canvas.unsqueeze(0)
    # st()
    rgb_canvas = 0.5*rgb_canvas + 0.5*image_cpu.unsqueeze(0)
    return rgb_canvas


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_new = tensor.clone()
        for t, m, s in zip(tensor_new, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_new


class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)

    def forward(self, x):
        x_prev = x
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(x, dim=1))
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        slots = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))
        return slots, dots




class SlotCon(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.hungarian_matcher = HungarianMatcher()
        self.cross_entropy = nn.CrossEntropyLoss()
        # st()
        self.args = args

        self.ready_classifier = False

        if args.arch == 'resnet50_pretrained_classification':
            self.ready_classifier = True
        
        

        self.unnormalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.global_steps = 0

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048

        self.start_acc_mean_q = 0.0
        self.end_acc_mean_q = 0.0        
        self.start_acc_mean_k = 0.0
        self.end_acc_mean_k = 0.0

        dataset_name = 'coco_2017_train_panoptic_with_sem_seg'
        dataset_name = 'coco_2017_val_panoptic_with_sem_seg'        

        self.evaluator = COCOPanopticEvaluatorExampleBased( dataset_name, './output/inference')

        # if args.sl_layer:
        #     self.encoder_q = encoder(head_type='second_last')
        #     self.encoder_k = encoder(head_type='second_last')
        # else:
        # st()

        cfg = args.cfg
        model = build_model(cfg)
        # st()
        checkpointer = DetectionCheckpointer(model)
        # checkpointer.load(cfg.MODEL.WEIGHTS) 
        # st()

        self.encoder_q = model
        self.encoder_k = model
        self.mean_pq_score = []
        
        # self.encoder_q = encoder(head_type='early_return')
        # self.encoder_k = encoder(head_type='early_return')

        # st()


        # st()


        if args.do_seg_class:
            self.class_predict_q = nn.Sequential(nn.Linear(self.dim_out, 1024), nn.ReLU(), nn.Linear(1024, 134))
            self.class_loss = nn.CrossEntropyLoss(ignore_index=-1)            


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        self.group_loss_weight = args.group_loss_weight
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
            
        self.projector_q = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.num_prototypes = args.num_prototypes
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))
        self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor_slot)
            
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  

        if self.args.max_pool_classifier:
            for param_q, param_k in zip(self.class_predict_q.parameters(), self.class_predict_k.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)




    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        # st()
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def set_means(self, start_acc_q, end_acc_q, start_acc_k, end_acc_k):
        # st()
        self.start_acc_mean_q = start_acc_q
        self.end_acc_mean_q = end_acc_q
        self.start_acc_mean_k = start_acc_k
        self.end_acc_mean_k = end_acc_k

    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)
        # st()

        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = concat_all_gather(mask_k.view(-1))
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1
        return F.cross_entropy(logits, labels) * (2 * tau)


    def semantic_inference(self, mask_cls, mask_pred):
        st()
     



    def forward(self, input, is_test=False):
        # st()
        if is_test:
            vis_dict = {}
            prediction_enc_q, prediction_enc_k = (self.encoder_q(input),self.encoder_k(input))
            # st()

            # st()
            self.resize = torchvision.transforms.Resize([ input[0]['height'] , input[0]['width'] ])
            image_vis = self.resize(input[0]['image'].to(torch.uint8))
            image_vis = image_vis.permute(1,2,0).cpu().numpy()
            metadata = input[0]['metadata']
            # st()

            vis_dict['input_image'] = wandb.Image(image_vis[None], caption="input_image") 
            self.global_steps += 1

            visualizer = Visualizer(image_vis, metadata, instance_mode=True)
            if "panoptic_seg" in prediction_enc_q[0]:
                panoptic_seg, segments_info = prediction_enc_q[0]["panoptic_seg"]
                # st()
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.cpu(), segments_info
                )
                vis_out = vis_output.get_image()   
                vis_dict['out_panoptic_seg'] = wandb.Image(vis_out[None], caption= 'panoptic_seg')
            # st()
            self.evaluator.reset()
            self.evaluator.process(input, prediction_enc_q)
            # st()
            results = self.evaluator.evaluate()
            self.mean_pq_score.append(results['panoptic_seg']['PQ'])
            print('PQ mean',torch.tensor(self.mean_pq_score).mean())
            st()

            # for index in range(len(input)):
            #     self.evaluator.process(input[index], prediction_enc_q[index])



            if False:
                image_vis = self.unnormalize(image)
                image_vis_img_1 = wandb.Image(image_vis[:1], caption="input_image")
                vis_dict['test_input_image_1'] = image_vis_img_1

                x1= self.projector_q(enc_q)
                y1 = self.projector_k(enc_k)

                (q1, score_q1)= self.grouping_q(x1)
                (k1, score_k1)= self.grouping_k(y1)        
                # st()    
                mask_vis_q1 = summ_instance_masks(score_q1[0],image_vis[0],pred=True)
                mask_vis_k1 = summ_instance_masks(score_k1[0],image_vis[0],pred=True)
                mask_vis_q1_img = wandb.Image(mask_vis_q1, caption="pred_mask_new")
                vis_dict['test_pred_mask_new_q1'] = mask_vis_q1_img
                mask_vis_k1_img = wandb.Image(mask_vis_k1, caption="pred_mask_new")
                vis_dict['test_pred_mask_new_k1'] = mask_vis_k1_img
                
                score_q1_ = score_q1.flatten(2,3)
                score_k1_ = score_k1.flatten(2,3)
                # st()

                mask_vis_1 = summ_instance_masks(masks_1[0],image_vis[0],pred=False)
                mask_vis_1_img = wandb.Image(mask_vis_1, caption="gt_mask_new_1")
                vis_dict['test_gt_mask_new_1'] = mask_vis_1_img     



                if self.args.do_tta:
                    vis_dict['start_acc_mean_q'] = self.start_acc_mean_q
                    vis_dict['end_acc_mean_q'] = self.end_acc_mean_q            
                    vis_dict['start_acc_mean_k'] = self.start_acc_mean_k
                    vis_dict['end_acc_mean_k'] = self.end_acc_mean_k  

                self.global_steps += 1
                vis_dict['test_total_loss'] = 0.0

            # loss = ari_score_q1
            return vis_dict
        else:
            input_0 = copy.deepcopy(input)
            input_1 = copy.deepcopy(input)
            input_keys = ['instances', 'image', 'flags', 'coords']

            for input_idx in range(len(input_0)):
                for key_val  in input_keys:
                    assert len(input_0[input_idx][key_val]) == 2
                    input_0[input_idx][key_val] = input_0[input_idx][key_val][0]


            for input_idx in range(len(input_1)):
                for key_val  in input_keys:
                    assert len(input_1[input_idx][key_val]) == 2
                    input_1[input_idx][key_val] = input_1[input_idx][key_val][1]

            # inputs_1 = inputs['image']
            # crops, coords, flags, masks = input

            vis_dict = {}

            # if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
            #     crops_vis_0 = self.unnormalize(crops[0])
            #     crops_vis_img_1 = wandb.Image(crops_vis_0[:1], caption="input_image")
            #     vis_dict['input_image_1'] = crops_vis_img_1

            #     crops_vis_1 = self.unnormalize(crops[1])
            #     crops_vis_img_2 = wandb.Image(crops_vis_1[:1], caption="input_image")
            #     vis_dict['input_image_2'] = crops_vis_img_2

            #     vis_dict['class_name'] = wandb.Html(class_str[0])
            loss_enc_q_0, loss_enc_q_1 = (self.encoder_q(input_0, get_ssl_features = True),self.encoder_q(input_1, get_ssl_features = True))

            st()
            if self.args.do_ssl:
                x1, x2 = self.projector_q(enc_q_0), self.projector_q(enc_q_1)            
                with torch.no_grad():  # no gradient to keys
                    if not (self.args.do_tta or self.args.fine_tune):
                        self._momentum_update_key_encoder()  # update the key encoder
                    
                    enc_k_0,enc_k_1 = (self.encoder_k(input_0),self.encoder_k(input_1))

                    y1, y2 = self.projector_k(enc_k_0), self.projector_k(enc_k_1)

                (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)

                score_q1_ = score_q1.flatten(2,3)
                score_q2_ = score_q2.flatten(2,3)

                # SSL loss
                q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])

                with torch.no_grad():
                    (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
                    k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
                
                cont_loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
                    + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))
                
                if not (self.args.do_tta or self.args.fine_tune):
                    self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))


                cont_loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
                    + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)



                if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
                    mask_vis_1 = summ_instance_masks(score_q1[0],crops_vis_0[0],pred=True)
                    mask_vis_2 = summ_instance_masks(score_q2[0],crops_vis_1[0],pred=True)
                    mask_vis_1_img = wandb.Image(mask_vis_1, caption="pred_mask_new")
                    vis_dict['pred_mask_new_1'] = mask_vis_1_img
                    mask_vis_2_img = wandb.Image(mask_vis_2, caption="pred_mask_new")
                    vis_dict['pred_mask_new_2'] = mask_vis_2_img


          
                vis_dict['total_loss'] = loss

            

            vis_dict['cont_loss'] = self.args.cont_weight * cont_loss

            self.global_steps += 1

        return loss, vis_dict
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SlotConEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        self.encoder_k = encoder(head_type='early_return')
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        for param_k in self.grouping_k.parameters():
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        with torch.no_grad():
            slots, probs = self.grouping_k(self.projector_k(self.encoder_k(x)))
            return probs