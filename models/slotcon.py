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
import matplotlib.pyplot as plt
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



class SlotCon(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.hungarian_matcher = HungarianMatcher()
        self.cross_entropy = nn.CrossEntropyLoss()
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

        self.encoder_name = encoder
        self.class_loss = nn.CrossEntropyLoss()

        self.encoder_q = encoder(head_type='early_return')

        if self.args.no_byol:
            self.encoder_k = self.encoder_q
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)            
        else:
            self.encoder_k = encoder(head_type='early_return')

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp

        self.num_prototypes = args.num_prototypes
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))
        self.all_centers = []


        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self):
        # st()
        if self.args.no_byol:
            self.encoder_q = self.encoder_name(head_type='early_return')
            self.encoder_k = self.encoder_q
        else:
            self.encoder_q = self.encoder_name(head_type='early_return')
            self.encoder_k = self.encoder_name(head_type='early_return') 
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            self.center = torch.zeros_like(self.center)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        # print('hello')
        # change_batchnorm_attr(self.encoder_k)
        # st()
        # print('hello')
        # self.encoder_k.eval()               
        # self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if self.args.update_teacher_tta and self.args.do_tta:
            momentum = self.teacher_momentum
        else:
            momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        # momentum will range between 0.99 (self.teacher_momentum) to 1
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        # for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
        #     param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        # for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
        #     param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  
        # if self.args.max_pool_classifier:
        #     for param_q, param_k in zip(self.class_predict_q.parameters(), self.class_predict_k.parameters()):
        #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)




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
        k = F.softmax(k / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def self_distill_probs(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        # k = F.softmax(k / self.teacher_temp, dim=-1)
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


     

    def visualize_plot(self,qs_aligned,ks_aligned,centers,class_label,ignore_center=False):

        score_aligned_q_probs = F.softmax(qs_aligned / self.student_temp, dim=-1).mean(0).clone().cpu().detach().numpy()
        score_aligned_q_pred_class = np.argmax(score_aligned_q_probs)
        plt.bar(np.arange(1000),score_aligned_q_probs)
        plt.title(f"predictied class; {score_aligned_q_pred_class},  gt class: {class_label}")
        plt.savefig(f"dump_plot_qs.png")
        plt.clf()
        plt.close()

        score_aligned_k_probs = F.softmax((ks_aligned) / self.teacher_temp, dim=-1).mean(0).clone().cpu().detach().numpy()
        score_aligned_k_pred_class = np.argmax(score_aligned_k_probs)
        plt.bar(np.arange(1000),score_aligned_k_probs)
        plt.title(f"predictied class; {score_aligned_k_pred_class},  gt class: {class_label}")
        plt.savefig(f"dump_plot_ks.png")
        plt.clf()
        plt.close()

        if not ignore_center:
            score_aligned_center_probs = F.softmax((centers) / self.teacher_temp, dim=-1).mean(0).clone().cpu().detach().numpy()
            score_aligned_center_pred_class = np.argmax(score_aligned_center_probs)
            plt.bar(np.arange(1000),score_aligned_center_probs)
            plt.title(f"predictied class; {score_aligned_center_pred_class},  gt class: {class_label}")
            plt.savefig(f"dump_plot_centers.png")
            plt.clf()
            plt.close()


        # st()
        

    def forward(self, input, is_test=False):
        if is_test:
            vis_dict = {}
            # st()
            image,class_labels,class_str = input
            image_vis = self.unnormalize(image)
            # print(class_str)
            if isinstance(class_str,str):
                vis_dict['test_class_name'] = wandb.Html(class_str[0])
            else:
                vis_dict['test_class_name'] = wandb.Html(class_str[0])

            image_vis_img_1 = wandb.Image(image_vis[:1], caption="input_image")
            vis_dict['test_input_image_1'] = image_vis_img_1

            enc_q,enc_k = (self.encoder_q(image),self.encoder_k(image))


            if self.args.do_tta:
                vis_dict['start_acc_mean_q'] = self.start_acc_mean_q
                vis_dict['end_acc_mean_q'] = self.end_acc_mean_q            
                vis_dict['start_acc_mean_k'] = self.start_acc_mean_k
                vis_dict['end_acc_mean_k'] = self.end_acc_mean_k  


            # B,_,_ = q1.shape

            class_labels_merged = class_labels
            total_num = torch.tensor(class_labels_merged.shape[0]).float()

            pred_class_q1,pred_class_k1 = (enc_q['fc'],enc_k['fc'])

            pred_idx_q1 = torch.argmax(pred_class_q1.squeeze(1),dim=-1)
            pred_idx_k1 = torch.argmax(pred_class_k1.squeeze(1),dim=-1)
            # st()
            if len(class_labels_merged.shape) ==1:
                pred_idx_q1_ = pred_idx_q1
                pred_idx_k1_ = pred_idx_k1
                similarity_q1 = pred_idx_q1_==class_labels_merged
                similarity_k1 = pred_idx_k1_==class_labels_merged
                vis_dict['k1_classification_acc_unnorm'] = similarity_k1
                vis_dict['q1_classification_acc_unnorm'] = similarity_q1                    
            else:
                pred_idx_q1_ = pred_idx_q1.unsqueeze(-1).repeat(1,class_labels_merged.shape[-1])
                pred_idx_k1_ = pred_idx_k1.unsqueeze(-1).repeat(1,class_labels_merged.shape[-1])
                similarity_q1 = pred_idx_q1_==class_labels_merged
                similarity_k1 = pred_idx_k1_==class_labels_merged
                similarity_k1 =similarity_k1.sum(-1).bool()
                similarity_q1 = similarity_q1.sum(-1).bool()
                vis_dict['k1_classification_acc_unnorm'] = similarity_k1
                vis_dict['q1_classification_acc_unnorm'] = similarity_q1

            
            num_correct_q1 = torch.sum(similarity_q1).float()

            acc_q1 = num_correct_q1/total_num
            
            vis_dict['q1_classification_acc'] = acc_q1
            
            num_correct_k1 = torch.sum(similarity_k1).float()

            acc_k1 = num_correct_k1/total_num

            vis_dict['k1_classification_acc'] = acc_k1
            # st()
  
            self.global_steps += 1

            vis_dict['test_total_loss'] = 0.0

  
            return vis_dict
        else:
            crops, class_labels,class_str = input
            

            vis_dict = {}

            if (self.global_steps % self.args.log_freq) ==0 and (not self.args.d):
                crops_vis_0 = self.unnormalize(crops[0])
                crops_vis_img_1 = wandb.Image(crops_vis_0[:1], caption="input_image")
                vis_dict['input_image_1'] = crops_vis_img_1

                crops_vis_1 = self.unnormalize(crops[1])
                crops_vis_img_2 = wandb.Image(crops_vis_1[:1], caption="input_image")
                vis_dict['input_image_2'] = crops_vis_img_2
    
                if isinstance(class_str,str):
                    vis_dict['test_class_name'] = wandb.Html(class_str)
                else:
                    vis_dict['test_class_name'] = wandb.Html(class_str[0])

            # st()
            enc_q_s = self.encoder_q(crops[0])

  
            
            if self.args.no_byol:
                enc_k_s = self.encoder_k(crops[1])
            else:
                with torch.no_grad():  # no gradient to keys
                    self.encoder_k.eval()
                    if not (self.args.do_tta or self.args.fine_tune) or self.args.update_teacher_tta:
                        self._momentum_update_key_encoder()  # update the key encoder
                    
                    enc_k_s = self.encoder_k(crops[1])

            qs_aligned = enc_q_s['fc']
            ks_aligned = enc_k_s['fc']



            # if self.global_steps == 1:
            #     st()

            # if self.global_steps == 23:
            #     st()
            # st()
            
            visualize_plot = False

            if not self.args.no_byol:
                if not (self.args.do_tta or self.args.fine_tune) or self.args.update_center_tta:
                    if self.args.merge_probs:
                        ks_aligned_prob = F.softmax(ks_aligned/self.teacher_temp,-1)
                        self.update_center_together(ks_aligned_prob)
                    else:
                        self.update_center_together(ks_aligned)



            if visualize_plot:
                if self.args.no_byol:
                    self.visualize_plot(qs_aligned, ks_aligned,self.center,class_labels[0], ignore_center=True)
                else:
                    self.visualize_plot(qs_aligned, ks_aligned,self.center,class_labels[0])
                
                vis_dict['qs_aligned_probs'] = wandb.Image("dump_plot_qs.png")
                vis_dict['ks_aligned_probs'] = wandb.Image("dump_plot_ks.png")
                if not self.args.no_byol:
                    vis_dict['center_probs'] = wandb.Image("dump_plot_centers.png")                

            # st()
            if self.args.center_loss:
                q1s = qs_aligned
                center_logits = torch.stack(self.all_centers).mean(0,keepdims=True).repeat(q1s.shape[0],1) 
                # st()
                if self.args.merge_probs:
                    cont_loss = self.self_distill_probs(q1s, center_logits)
                else:
                    cont_loss = self.self_distill(q1s, center_logits)
            else:
                cont_loss = self.self_distill(qs_aligned, ks_aligned)
            




            # st()
            # classification only loss
            if self.args.do_only_classification:
                class_labels_merged = class_labels
                pred_class = qs_aligned
                # class_loss = 0.0
                pred_idx = torch.argmax(pred_class.squeeze(1),dim=-1)

                if len(class_labels_merged.shape) ==1:
                    class_loss = self.class_loss(pred_class.squeeze(1),class_labels_merged)
                    pred_idx_ = pred_idx
                    similarity = pred_idx_==class_labels_merged
                    num_correct = torch.sum(similarity).float()
                else:
                    class_loss = self.class_loss(pred_class.squeeze(1),class_labels_merged[:,0])
                    pred_idx_ = pred_idx.unsqueeze(-1).repeat(1,class_labels_merged.shape[1])
                    similarity = pred_idx_==class_labels_merged
                    num_correct = torch.sum(similarity.sum(-1).bool()).float()
                # num_correct = torch.sum(pred_idx==class_labels_merged).float()
                total_num = torch.tensor(class_labels_merged.shape[0]).float()
                acc = num_correct/total_num
                vis_dict['classification_acc'] = acc

            # st()

            loss = (self.args.cont_weight * cont_loss) + (self.args.class_weight * class_loss )
            vis_dict['classification_loss'] = self.args.class_weight * class_loss
            vis_dict['cont_loss'] = self.args.cont_weight * cont_loss
            vis_dict['total_loss'] = loss
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
    def update_center_together(self, teacher_output):
        """
        Update center used for teacher output.
        """
        teacher_output_ = list(teacher_output.unbind(0))
        self.all_centers += teacher_output_
        # st()
        # batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # # ema update
        # self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

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