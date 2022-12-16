import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import matplotlib.pyplot as plt
import wandb
import numpy as np

from hungarian_matcher import HungarianMatcher
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
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


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
        self.args = args
        # st()
        self.unnormalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.global_steps = 0

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        self.encoder_q = encoder(head_type='early_return')
        self.encoder_k = encoder(head_type='early_return')

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

    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)

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

    def forward(self, input):
        crops, coords, flags, masks = input
        # st()
        vis_dict = {}
        # st()
        if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
            crops_vis = self.unnormalize(crops)
            crops_vis_img_1 = wandb.Image(crops_vis[0][:1], caption="input_image")
            vis_dict['input_image_1'] = crops_vis_img_1

            crops_vis_img_2 = wandb.Image(crops_vis[1][:1], caption="input_image")
            vis_dict['input_image_2'] = crops_vis_img_2
        # st()



        masks_1, masks_2 = masks[0], masks[1]

        x1, x2 = self.projector_q(self.encoder_q(crops[0])), self.projector_q(self.encoder_q(crops[1]))
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.projector_k(self.encoder_k(crops[0])), self.projector_k(self.encoder_k(crops[1]))


        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)


        # st()
        if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d): 
            mask_vis_1 = summ_instance_masks(score_q1[0],crops_vis[0][0],pred=True)
            mask_vis_2 = summ_instance_masks(score_q2[0],crops_vis[1][0],pred=True)
            mask_vis_1_img = wandb.Image(mask_vis_1, caption="pred_mask_new")
            vis_dict['pred_mask_new_1'] = mask_vis_1_img
            mask_vis_2_img = wandb.Image(mask_vis_2, caption="pred_mask_new")
            vis_dict['pred_mask_new_2'] = mask_vis_2_img


            mask_vis_1 = summ_instance_masks(masks_1[0],crops_vis[0][0],pred=False)
            mask_vis_2 = summ_instance_masks(masks_2[0],crops_vis[1][0],pred=False)
            mask_vis_1_img = wandb.Image(mask_vis_1, caption="gt_mask_new_1")
            vis_dict['gt_mask_new_1'] = mask_vis_1_img
            mask_vis_2_img = wandb.Image(mask_vis_2, caption="gt_mask_new_2")
            vis_dict['gt_mask_new_2'] = mask_vis_2_img            
        # find match
        score_q1_ = score_q1.flatten(2,3)
        score_q2_ = score_q2.flatten(2,3)

        masks_1_ = masks_1.flatten(2,3)
        masks_2_ = masks_2.flatten(2,3)

        # new_indices = self.hungarian_matcher(masks_1_,score_q1_)
        # masks_1_u_ = []
        # score_q1_u_ = []
        # for ind_ex, indices_ex in enumerate(new_indices):
        #     p_indices,gt_indices = indices_ex
        #     score_q1_u_.append(score_q1_[ind_ex,gt_indices])
        #     masks_1_u_.append(masks_1_[ind_ex,p_indices])
        # masks_1_u = torch.stack(masks_1_u_)
        # score_q1_u = torch.stack(score_q1_u_)
        # ce_loss_1 = self.cross_entropy(score_q1_u, masks_1_u)


        # new_indices = self.hungarian_matcher(masks_2_,score_q2_)
        # masks_2_u_ = []
        # score_q2_u_ = []
        # for ind_ex, indices_ex in enumerate(new_indices):
        #     p_indices,gt_indices = indices_ex
        #     score_q2_u_.append(score_q2_[ind_ex,gt_indices])
        #     masks_2_u_.append(masks_2_[ind_ex,p_indices])
        # masks_2_u = torch.stack(masks_2_u_)
        # score_q2_u = torch.stack(score_q2_u_)
        # ce_loss_2 = self.cross_entropy(score_q2_u, masks_2_u)
        # st()

        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
        
        loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))

        self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))

        loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
              + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)

        vis_dict['total_loss'] = loss
        if not self.args.d:
            wandb.log(vis_dict,step=self.global_steps)
        self.global_steps += 1

        return loss
    
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