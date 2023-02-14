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

from timm.models.vision_transformer import PatchEmbed, Block


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
        self.args = args

        self.ready_classifier = False

        if args.arch == 'resnet50_pretrained_classification':
            self.ready_classifier = True

        # st()
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
        # st()
        if args.sl_layer:
            self.encoder_q = encoder(head_type='second_last')
            self.encoder_k = encoder(head_type='second_last')
        else:
            self.encoder_q = encoder(head_type='early_return')
            self.encoder_k = encoder(head_type='early_return')

        # st()


        # st()
        if args.do_only_classification:
            if args.vit_probing:
                classifier_embed_dim = 768
                num_classes = 1000
                self.classifier_cls_token = nn.Parameter(torch.randn(1, 1, self.num_channels), requires_grad=True)
                self.classifier_embed = nn.Linear(self.num_channels, classifier_embed_dim, bias=True)
                self.classifier_pos_embed = nn.Parameter(torch.randn(1, 50, classifier_embed_dim), requires_grad=True)  # fixed sin-cos embedding

                dpr = [x.item() for x in torch.linspace(0, 0.0, 12)]
                self.classifier_blocks = nn.ModuleList([
                        Block(classifier_embed_dim, 8, 4, qkv_bias=True, norm_layer=nn.LayerNorm, drop_path=dpr[i])
                    for i in range(12)])

                self.classifier_norm = nn.LayerNorm(classifier_embed_dim)
                self.classifier_pred = nn.Linear(classifier_embed_dim, num_classes)
            elif self.ready_classifier:
                pass
            elif self.args.max_pool_classifier:
                self.class_predict_k = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Flatten(), nn.Linear(2048, 1000))
                self.class_predict_q = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Flatten(), nn.Linear(2048, 1000))
                for param_q, param_k in zip(self.class_predict_q.parameters(), self.class_predict_k.parameters()):
                    param_k.data.copy_(param_q.data)  # initialize
                    param_k.requires_grad = False  # not update by gradient
                nn.SyncBatchNorm.convert_sync_batchnorm(self.class_predict_q)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.class_predict_k)
            else:
                self.mha = nn.MultiheadAttention(1024, 1, batch_first=True)
                self.key = nn.Linear(self.dim_out, 1024)
                self.value = nn.Linear(self.dim_out, 1024)
                self.query_embed = nn.Parameter(torch.randn(1, 1, 1024))
                self.class_predict = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1000))
            self.class_loss = nn.CrossEntropyLoss()



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
        if is_test:
            vis_dict = {}
            image,masks_1,class_labels = input
            image_vis = self.unnormalize(image)
            image_vis_img_1 = wandb.Image(image_vis[:1], caption="input_image")
            vis_dict['test_input_image_1'] = image_vis_img_1
            enc_q,enc_k = (self.encoder_q(image),self.encoder_k(image))
            if self.ready_classifier:
                x1, y1 = self.projector_q(enc_q['layer4']), self.projector_k(enc_k['layer4'])
            else:
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
            if not self.args.do_only_classification:
                mask_vis_1 = summ_instance_masks(masks_1[0],image_vis[0],pred=False)
                mask_vis_1_img = wandb.Image(mask_vis_1, caption="gt_mask_new_1")
                vis_dict['test_gt_mask_new_1'] = mask_vis_1_img     
                masks_1_ = masks_1.flatten(2,3)
                ari_score_q1 = ari.adjusted_rand_index(masks_1_.permute(0,2,1).cpu(),score_q1_.permute(0,2,1).cpu()).mean()
                ari_score_k1 = ari.adjusted_rand_index(masks_1_.permute(0,2,1).cpu(),score_k1_.permute(0,2,1).cpu()).mean()

                vis_dict['test_ari_score_q1'] = ari_score_q1     
                vis_dict['test_ari_score_k1'] = ari_score_k1


            if self.args.do_tta:
                vis_dict['start_acc_mean_q'] = self.start_acc_mean_q
                vis_dict['end_acc_mean_q'] = self.end_acc_mean_q            
                vis_dict['start_acc_mean_k'] = self.start_acc_mean_k
                vis_dict['end_acc_mean_k'] = self.end_acc_mean_k  
            # st()

            if self.args.do_seg_class:
                # st()
                q1_pred_class = self.class_predict(q1)
                k1_pred_class = self.class_predict(k1)
                # q1_gt_labels = class_labels
                # k1_gt_labels = class_labels
                class_labels_valid = (class_labels != -1).float()
                
                q1_pred_idx = torch.argmax(q1_pred_class.squeeze(1),dim=-1)
                q1_correct = (q1_pred_idx==class_labels).float()
                q1_num_correct = torch.sum(class_labels_valid*q1_correct).float()
                q1_total_num = torch.sum(class_labels_valid).float()
                q1_acc = q1_num_correct/q1_total_num
                vis_dict['q1_classification_acc'] = q1_acc

                k1_pred_idx = torch.argmax(k1_pred_class.squeeze(1),dim=-1)
                k1_correct = (k1_pred_idx==class_labels).float()
                k1_num_correct = torch.sum(class_labels_valid*k1_correct).float()
                k1_total_num = torch.sum(class_labels_valid).float()
                k1_acc = k1_num_correct/k1_total_num
                vis_dict['k1_classification_acc'] = k1_acc

                # st()
            if self.args.do_only_classification:
                B,_,_ = q1.shape
                # qs = torch.cat([q1,q2],0)
                class_labels_merged = class_labels
                total_num = torch.tensor(class_labels_merged.shape[0]).float()
                # st()
                if self.ready_classifier:
                    pred_class_q1,pred_class_k1 = (enc_q['fc'],enc_k['fc'])
                elif  self.args.vit_probing:
                    B_split = enc_q.shape[0]
                    enc_qk = torch.cat([enc_q,enc_k],0)
                    enc_qk_ = enc_qk.flatten(2,3).permute(0,2,1)
                    classifier_cls_token_ = self.classifier_cls_token.repeat(enc_qk_.shape[0],1,1)
                    enc_qk_cls = torch.cat([classifier_cls_token_,enc_qk_],1)
                    enc_qk_cls = self.classifier_embed(enc_qk_cls)
                    enc_qk_cls = enc_qk_cls + self.classifier_pos_embed


                    for blk in self.classifier_blocks:
                        enc_qk_cls = blk(enc_qk_cls)

                    enc_qk_cls = self.classifier_norm(enc_qk_cls)

                    enc_cls = enc_qk_cls[:, :1, :]
                    pred_class = self.classifier_pred(enc_cls)[:, 0]
                    pred_class_q1 = pred_class[:B_split]
                    pred_class_k1 = pred_class[B_split:]
                elif self.args.max_pool_classifier:
                    pred_class_q1 = self.class_predict_q(enc_q)
                    pred_class_k1 = self.class_predict_k(enc_k)
                else:
                    query_embed = self.query_embed.repeat(B,1,1)
                    cls_token_q1,_ = self.mha(query_embed,self.key(q1),self.value(q1))
                    pred_class_q1 = self.class_predict(cls_token_q1)
                    cls_token_k1,_ = self.mha(query_embed,self.key(k1),self.value(k1))
                    pred_class_k1 = self.class_predict(cls_token_k1)


                pred_idx_q1 = torch.argmax(pred_class_q1.squeeze(1),dim=-1)
                num_correct_q1 = torch.sum(pred_idx_q1==class_labels_merged).float()
                acc_q1 = num_correct_q1/total_num
                vis_dict['q1_classification_acc'] = acc_q1

                pred_idx_k1 = torch.argmax(pred_class_k1.squeeze(1),dim=-1)
                num_correct_k1 = torch.sum(pred_idx_k1==class_labels_merged).float()
                acc_k1 = num_correct_k1/total_num
                vis_dict['k1_classification_acc'] = acc_k1
                # print(acc_q1,acc_k1)
                # st():

            # st()
            self.global_steps += 1

            vis_dict['test_total_loss'] = 0.0

            # loss = ari_score_q1
            return vis_dict
        else:
            crops, coords, flags, masks, class_labels,class_str = input

            vis_dict = {}

            if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
                crops_vis_0 = self.unnormalize(crops[0])
                crops_vis_img_1 = wandb.Image(crops_vis_0[:1], caption="input_image")
                vis_dict['input_image_1'] = crops_vis_img_1

                crops_vis_1 = self.unnormalize(crops[1])
                crops_vis_img_2 = wandb.Image(crops_vis_1[:1], caption="input_image")
                vis_dict['input_image_2'] = crops_vis_img_2

                vis_dict['class_name'] = wandb.Html(class_str[0])

            # st()
            enc_q_0, enc_q_1 = (self.encoder_q(crops[0]),self.encoder_q(crops[1]))
            if self.ready_classifier:
                x1, x2 = self.projector_q(enc_q_0['layer4']), self.projector_q(enc_q_1['layer4'])
            else:
                x1, x2 = self.projector_q(enc_q_0), self.projector_q(enc_q_1)
            
            with torch.no_grad():  # no gradient to keys
                if not (self.args.do_tta or self.args.fine_tune):
                    self._momentum_update_key_encoder()  # update the key encoder
                
                enc_k_0,enc_k_1 = (self.encoder_k(crops[0]),self.encoder_k(crops[1]))

                if self.ready_classifier:
                    y1, y2 = self.projector_k(enc_k_0['layer4']), self.projector_k(enc_k_1['layer4'])                    
                else:
                    y1, y2 = self.projector_k(enc_k_0), self.projector_k(enc_k_1)


            (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)


            score_q1_ = score_q1.flatten(2,3)
            score_q2_ = score_q2.flatten(2,3)
            # st()

            # self.mha(self.)


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


            # st()


            if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
                mask_vis_1 = summ_instance_masks(score_q1[0],crops_vis_0[0],pred=True)
                mask_vis_2 = summ_instance_masks(score_q2[0],crops_vis_1[0],pred=True)
                mask_vis_1_img = wandb.Image(mask_vis_1, caption="pred_mask_new")
                vis_dict['pred_mask_new_1'] = mask_vis_1_img
                mask_vis_2_img = wandb.Image(mask_vis_2, caption="pred_mask_new")
                vis_dict['pred_mask_new_2'] = mask_vis_2_img


            if self.args.do_seg_class:
                # qs = torch.cat([q1,q2],0)
                q1_pred_class = self.class_predict(q1)
                q2_pred_class = self.class_predict(q2)
                q1_gt_labels = class_labels
                q2_gt_labels = class_labels

            # segmentation loss
            if not self.args.do_only_classification:
                masks_1, masks_2 = masks[0], masks[1]

                if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
                    mask_vis_1 = summ_instance_masks(masks_1[0],crops_vis_0[0],pred=False)
                    mask_vis_2 = summ_instance_masks(masks_2[0],crops_vis_1[0],pred=False)
                    mask_vis_1_img = wandb.Image(mask_vis_1, caption="gt_mask_new_1")
                    vis_dict['gt_mask_new_1'] = mask_vis_1_img
                    mask_vis_2_img = wandb.Image(mask_vis_2, caption="gt_mask_new_2")
                    vis_dict['gt_mask_new_2'] = mask_vis_2_img            

                
                masks_1_ = masks_1.flatten(2,3)
                masks_2_ = masks_2.flatten(2,3)
                # st()
                ari_score_1 = ari.adjusted_rand_index(masks_1_.permute(0,2,1).cpu(),score_q1_.permute(0,2,1).cpu())
                ari_score_2 = ari.adjusted_rand_index(masks_2_.permute(0,2,1).cpu(),score_q2_.permute(0,2,1).cpu())
                # print(ari_score_1, ari_score_2)
                ari_score = ((ari_score_1 + ari_score_2)/2.0).mean()

                if self.args.seg_weight >0.0:
                    # score_q1_.permute(0,2,1).unique() masks_1_.permute(0,2,1).unique() masks_1_.permute(0,2,1).sum(-1).unique()
                    new_indices = self.hungarian_matcher(score_q1_,masks_1_, do_softmax=True)
                    masks_1_u_ = []
                    score_q1_u_ = []
                    
                    class_1_u_ = []
                    pred_class_1_u_ = []

                    for ind_ex, indices_ex in enumerate(new_indices):
                        p_indices,gt_indices = indices_ex
                        masks_1_u_.append(masks_1_[ind_ex,gt_indices])
                        score_q1_u_.append(score_q1_[ind_ex,p_indices])

                        if self.args.do_seg_class:
                            class_1_u_.append(q1_gt_labels[ind_ex,gt_indices])
                            pred_class_1_u_.append(q1_pred_class[ind_ex,p_indices])



                    masks_1_u = torch.stack(masks_1_u_)
                    score_q1_u = torch.stack(score_q1_u_)

                    class_1_u = torch.stack(class_1_u_)
                    pred_class_1_u = torch.stack(pred_class_1_u_)

                    ce_loss_1 = self.cross_entropy(score_q1_u, masks_1_u)
                    # st()

                    new_indices = self.hungarian_matcher(score_q2_,masks_2_, do_softmax=True)
                    masks_2_u_ = []
                    score_q2_u_ = []

                    class_2_u_ = []
                    pred_class_2_u_ = []

                    for ind_ex, indices_ex in enumerate(new_indices):
                        p_indices,gt_indices = indices_ex
                        masks_2_u_.append(masks_2_[ind_ex,gt_indices])
                        score_q2_u_.append(score_q2_[ind_ex,p_indices])

                        if self.args.do_seg_class:
                            class_2_u_.append(q2_gt_labels[ind_ex,gt_indices])
                            pred_class_2_u_.append(q2_pred_class[ind_ex,p_indices])


                    masks_2_u = torch.stack(masks_2_u_)
                    score_q2_u = torch.stack(score_q2_u_)

                    if self.args.do_seg_class:
                        class_2_u = torch.stack(class_2_u_)
                        pred_class_2_u = torch.stack(pred_class_2_u_)

                    # st()
                    ce_loss_2 = self.cross_entropy(score_q2_u, masks_2_u)

                    seg_supervised_loss = ce_loss_1 + ce_loss_2
                    # print(ce_loss_1,ce_loss_2,score_q2_u.mean())

                else:
                    seg_supervised_loss = 0.0
            else:
                seg_supervised_loss = 0.0
                ari_score = 0.0

            if self.args.do_seg_class:
                pred_classes = torch.cat([pred_class_1_u,pred_class_2_u],0)
                pred_classes = pred_classes.reshape(-1,pred_classes.shape[-1])

                class_labels_merged = torch.cat([class_1_u,class_2_u],0).reshape(-1)  
                # st()
                class_loss = self.class_loss(pred_classes.squeeze(1),class_labels_merged.long())

                pred_idx = torch.argmax(pred_classes.squeeze(1),dim=-1)
                class_labels_valid = (class_labels_merged != -1).float()
                correct = (pred_idx==class_labels_merged).float()

                num_correct = torch.sum(class_labels_valid*correct).float()
                total_num = torch.sum(class_labels_valid).float()
                acc = num_correct/total_num
                vis_dict['classification_acc'] = acc
                # st()
                # vis_dict['predicted_segments'] = acc

                # st()


            # classification only loss
            if self.args.do_only_classification:
                class_labels_merged = torch.cat([class_labels,class_labels],0)
                # st()
                if self.ready_classifier:
                    pred_class = torch.cat([enc_q_0['fc'],enc_q_1['fc']], 0)
                elif self.args.vit_probing:
                    enc_qs = torch.cat([enc_q_0,enc_q_1],0)
                    enc_qs_ = enc_qs.flatten(2,3).permute(0,2,1)
                    classifier_cls_token_ = self.classifier_cls_token.repeat(enc_qs_.shape[0],1,1)
                    enc_qs_cls = torch.cat([classifier_cls_token_,enc_qs_],1)
                    enc_qs_cls = self.classifier_embed(enc_qs_cls)
                    enc_qs_cls = enc_qs_cls + self.classifier_pos_embed

                    # st()
                    for blk in self.classifier_blocks:
                        enc_qs_cls = blk(enc_qs_cls)
                    # st()
                    enc_qs_cls = self.classifier_norm(enc_qs_cls)
                    # st()
                    enc_cls = enc_qs_cls[:, :1, :]
                    pred_class = self.classifier_pred(enc_cls)[:, 0]
                    # st()
                elif self.args.max_pool_classifier:
                    enc_qs = torch.cat([enc_q_0,enc_q_1],0)
                    pred_class = self.class_predict_q(enc_qs)
                else:
                    B,_,_ = q1.shape
                    qs = torch.cat([q1,q2],0)
                    query_embed = self.query_embed.repeat(B*2,1,1)
                    cls_token,_ = self.mha(query_embed,self.key(qs),self.value(qs))
                    pred_class = self.class_predict(cls_token)
                class_loss = self.class_loss(pred_class.squeeze(1),class_labels_merged)
                pred_idx = torch.argmax(pred_class.squeeze(1),dim=-1)
                num_correct = torch.sum(pred_idx==class_labels_merged).float()
                total_num = torch.tensor(class_labels_merged.shape[0]).float()
                acc = num_correct/total_num
                vis_dict['classification_acc'] = acc
                # st()

            # st()
            loss = (self.args.seg_weight * seg_supervised_loss) + (self.args.cont_weight * cont_loss) + (self.args.class_weight * class_loss )

            
            vis_dict['classification_loss'] = self.args.class_weight * class_loss
            vis_dict['cont_loss'] = self.args.cont_weight * cont_loss
            vis_dict['seg_loss'] = self.args.seg_weight * seg_supervised_loss
            vis_dict['total_loss'] = loss
            vis_dict['ari_score'] = ari_score

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