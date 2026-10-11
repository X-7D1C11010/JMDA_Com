import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.classifier import SimpleFC2
from models.networks.discriminator import DomainDiscriminator
from models.networks.msa import  Bert, APViT_video, Wavlm

from models.myfunc import entropy_re
from models.func import ClassEmbedding, SupConLoss
from models.opt import pte, pte_qp
from models.coralutils.coral import coral

import numpy as np
from scipy.stats import entropy
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import copy


from models.min_norm_solvers import  gradient_normalizers


def p2oh(prob_matrix):
    # Find the indices of the maximum probabilities for each row
    max_indices = torch.argmax(prob_matrix, dim=1)

    # Create a one-hot encoded matrix
    one_hot_matrix = torch.zeros_like(prob_matrix)
    one_hot_matrix.scatter_(1, max_indices.unsqueeze(1), 1)

    return one_hot_matrix

def index_rely(matrix, threshold=3):
    # Find the rows where any element is greater than the threshold
    large_elements = matrix >= threshold
    rows_with_large_elements = torch.any(large_elements, dim=1)

    # Get the indices of these rows
    row_indices = torch.nonzero(rows_with_large_elements).squeeze(1)

    return row_indices


class BoomdaModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.opt = opt
        self.weight = opt.weight
        self.lora_r = 128
        self.is_amp = 1
        disc_coe = 1 #self.weight
        self.scaler = GradScaler()
        self.use_cvx = 0
        self.use_cvx_qp = 0
        self.use_cfs = 0
        self.use_full = 1
        self.pl_num = 0
    
        self.modality = opt.modality
        all_len = len(self.modality) + 1
        scalet = [1] * 4
        for i, t in enumerate('AVL'):
            if t not in self.modality:
                scalet[i] = 0
        self.scale = np.array(scalet) / all_len
        #self.weight = 0
        self.d_f = 32
        self.error_pte = []
        self.error_pte2 = []
        self.disc_test = opt.disc_test
        self.loss_names = ['CE', 'CE_t', 'CE_tpl', 'CE_A', 'CE_V', 'CE_L', 'coralall', 'coral_a', 'coral_v', 'coral_l', 'coral']
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        # self.netC = FcClassifier(cls_input_size, [], output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netC = SimpleFC2(input_dim=cls_input_size, hidden=[128], output_dim=opt.output_dim, d_f=self.d_f)

        self.netembedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=128).to(self.device)

        self.clsemddim = 32
        self.netembedding2 = ClassEmbedding(dim=self.clsemddim, n=100)
        self.netdiscCom = DomainDiscriminator(cls_input_size, disc_coe)
        self.model_names.append('discCom')
        
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.model_names.append('CA')
            self.netCA = SimpleFC2(input_dim=opt.embd_size_a, hidden=[64], output_dim=opt.output_dim, d_f=self.d_f)

            self.netA = Wavlm(self.device, self.lora_r, opt.embd_size_a)
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.model_names.append('CL')
            self.netCL = SimpleFC2(input_dim=opt.embd_size_l, hidden=[128], output_dim=opt.output_dim, d_f=self.d_f)

            self.netL = Bert(self.device, self.lora_r, opt.embd_size_l)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.model_names.append('CV')
            self.netCV = SimpleFC2(input_dim=opt.embd_size_v, hidden=[64], output_dim=opt.output_dim, d_f=self.d_f)

            self.netV = APViT_video(self.device, self.lora_r, opt.embd_size_v)
            
        if self.isTrain:

            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_domdisc = torch.nn.CrossEntropyLoss() #torch.nn.BCELoss()   #
            self.criterion_supcon = SupConLoss()
            self.criterion_nll = torch.nn.NLLLoss()
            self.criterion_mse = torch.nn.MSELoss()
            
            ft_lr = 1e-5 #5e-5 #1e-4
            parameters = [
                *[
                    {'params': [param for param in getattr(self, 'net' + net).parameters() if net not in ['A', 'V', 'L']], 
                    'weight_decay': opt.weight_decay, 'lr':opt.lr} 
                  
                    for net in self.model_names
                ]]

            if 'A' in self.modality:
                acoustic_params = []
                other_params_A = []
                for name, param in self.netA.named_parameters():
                    if 'model_A' in name:  
                        acoustic_params.append(param)
                    else:
                        other_params_A.append(param)
                
                # Create parameter groups for optimizer
                parameters.append({'params': acoustic_params, 'lr': ft_lr})
                parameters.append({'params': other_params_A, 'lr': opt.lr})

            if 'V' in self.modality:
                visual_params = []
                other_params_V = []
                for name, param in self.netV.named_parameters():
                    if 'model_V' in name:  
                        visual_params.append(param)
                    else:
                        other_params_V.append(param)
                
                # Create parameter groups for optimizer
                parameters.append({'params': visual_params, 'lr': ft_lr})
                parameters.append({'params': other_params_V, 'lr': opt.lr})

            if 'L' in self.modality:
                lexical_params = []
                other_params_L = []
                for name, param in self.netL.named_parameters():
                    if 'model_L' in name:  
                        lexical_params.append(param)
                    else:
                        other_params_L.append(param)
                
                # Create parameter groups for optimizer
                parameters.append({'params': lexical_params, 'lr': ft_lr})
                parameters.append({'params': other_params_L, 'lr': opt.lr})
            
                              
            
            self.optimizer = torch.optim.Adam(parameters, betas=(opt.beta1, 0.999))
            self.output_dim = opt.output_dim
            self.optimizers.append(self.optimizer)


        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input, input_t=None, alpha=None):
        """F
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        if self.isTrain or (self.disc_test and input_t != None):
            if 'A' in self.modality:
                self.acoustic = input['audio']
                self.acoustic_t = input_t['audio']

            if 'L' in self.modality:
                self.lexical = input['text']
                self.lexical_t = input_t['text']
   
            if 'V' in self.modality:
                self.visual = input['vision']
                self.visual_t = input_t['vision']

            self.label = input['label'].to(self.device)
            self.label_t = input_t['label'].to(self.device)
            self.alpha = alpha


            self.bs_s = self.label.shape[0]
            self.bs_t = self.label_t.shape[0]
            domain_source_labels = torch.zeros(self.label.shape[0]).type(torch.LongTensor) 
            domain_target_labels = torch.ones(self.label_t.shape[0]).type(torch.LongTensor)  
            self.domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()

        else:
            if 'A' in self.modality:
                self.acoustic = input['audio']
            if 'L' in self.modality:
                self.lexical = input['text']

            if 'V' in self.modality:
                self.visual = input['vision']
            
            self.label = input['label']
            


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.final_embd = []
        self.final_embd_target = []
        if self.isTrain:
            if 'A' in self.modality:
                self.feat_A = self.netA(**self.acoustic)
                self.final_embd.append(self.feat_A)

                self.feat_A_t = self.netA(**self.acoustic_t)
                self.final_embd_target.append(self.feat_A_t)
                
            if 'V' in self.modality:
                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)

                self.feat_V_t = self.netV(self.visual_t)
                self.final_embd_target.append(self.feat_V_t)

            if 'L' in self.modality:
                self.feat_L = self.netL(**self.lexical)              
                self.final_embd.append(self.feat_L)


                self.feat_L_t = self.netL(**self.lexical_t)
                self.final_embd_target.append(self.feat_L_t)

        else:
            if 'A' in self.modality:

                self.feat_A = self.netA(**self.acoustic)
                self.final_embd.append(self.feat_A)
                self.logits_A, _ = self.netCA(self.feat_A)
                self.pred_A = F.softmax(self.logits_A, dim=-1)
            
            if 'V' in self.modality:
                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)
                self.logits_V, _ = self.netCV(self.feat_V)
                self.pred_V = F.softmax(self.logits_V, dim=-1)

            if 'L' in self.modality:
                self.feat_L= self.netL(**self.lexical)
                self.final_embd.append(self.feat_L)
                
                self.logits_L, _ = self.netCL(self.feat_L)
                self.pred_L = F.softmax(self.logits_L, dim=-1)
        
            # get model outputs
            self.feat = torch.cat(self.final_embd, dim=-1)
            self.logits, self.fall = self.netC(self.feat)
            self.pred = F.softmax(self.logits, dim=-1)


        if self.isTrain:

            bs_s = self.bs_s
            bs_t = self.bs_t
            bs_a = bs_s + bs_t
            self.feat = torch.cat(self.final_embd, dim=-1)
            self.feat_t = torch.cat(self.final_embd_target, dim=-1)
            
            self.feat_all = torch.cat([self.feat, self.feat_t], dim=0)
            self.feat_all.register_hook(self.save_gradients)

            
            self.feat_r = self.feat_all[0: bs_s, :]
            self.feat_t_r = self.feat_all[bs_s: bs_a, :]
            
            self.logits, self.fall = self.netC(self.feat_r)
            self.pred = F.softmax(self.logits, dim=-1)

            self.logits_t, self.fall_t = self.netC(self.feat_t_r)
            self.pred_t = F.softmax(self.logits_t, dim=-1)  
            self.pred_toh = p2oh(self.pred_t.detach())
            self.pred_tf = 1.5 * self.pred_toh

            self.entropy_t = entropy(self.pred_t.T.cpu().detach().numpy())
            self.pred_t_ave = self.pred_t.mean(dim=0)

            dim_a = self.opt.embd_size_a * int("A" in self.modality)
            
            dim_av = self.opt.embd_size_a * int("A" in self.modality) + \
                             self.opt.embd_size_v * int("V" in self.modality)

            dim_avl = self.opt.embd_size_a * int("A" in self.modality) + \
                             self.opt.embd_size_v * int("V" in self.modality) + \
                             self.opt.embd_size_l * int("L" in self.modality)

            if 'A' in self.modality:

                self.feat_A_r = self.feat_all[0:bs_s, 0:dim_a]
                self.feat_A_t_r = self.feat_all[bs_s:bs_a, 0:dim_a]



                self.logits_A, self.fa = self.netCA(self.feat_A_r)
                self.pred_A = F.softmax(self.logits_A, dim=-1)

                self.logits_A_t, self.fa_t = self.netCA(self.feat_A_t_r)
                self.pred_A_t = F.softmax(self.logits_A_t, dim=-1)
                self.pred_A_t_ave = self.pred_A_t.mean(dim=0)
                self.pred_A_toh = p2oh(self.pred_A_t.detach())
                self.pred_tf += self.pred_A_toh
            
            if 'V' in self.modality: 
                self.feat_V_r = self.feat_all[0:bs_s, dim_a:dim_av]
                self.feat_V_t_r = self.feat_all[bs_s:bs_a, dim_a:dim_av]

                self.logits_V, self.fv = self.netCV(self.feat_V_r)
                self.pred_V = F.softmax(self.logits_V, dim=-1)

                self.logits_V_t, self.fv_t = self.netCV(self.feat_V_t_r)
                self.pred_V_t = F.softmax(self.logits_V_t, dim=-1)
                self.pred_V_t_ave = self.pred_V_t.mean(dim=0)
                self.pred_V_toh = p2oh(self.pred_V_t.detach())
                self.pred_tf += self.pred_V_toh


            if 'L' in self.modality: 
                self.feat_L_r = self.feat_all[0:bs_s, dim_av: dim_avl]
                self.feat_L_t_r = self.feat_all[bs_s: bs_a, dim_av: dim_avl]


                self.logits_L, self.fl = self.netCL(self.feat_L_r)
                self.pred_L = F.softmax(self.logits_L, dim=-1)

                self.logits_L_t, self.fl_t = self.netCL(self.feat_L_t_r)
                self.pred_L_t = F.softmax(self.logits_L_t, dim=-1)
                self.pred_L_t_ave = self.pred_L_t.mean(dim=0)
                self.pred_L_toh = p2oh(self.pred_L_t.detach())
                self.pred_tf += self.pred_L_toh
            
            self.domainCom = self.netdiscCom(self.feat_all, self.alpha)
            self.index_pl = index_rely(self.pred_tf)
            self.pred_tf =  self.pred_tf.argmax(dim=1)
            self.pl = self.pred_tf[self.index_pl]
            if self.index_pl.shape[0] >=1:
                self.pl_num += self.index_pl.shape[0]
   

    def save_gradients(self, grad):
        self.feat_all_gradients = grad

    def weight_cal(self):
        loss_data = {}
        loss_data_m = {}
        grads_all = {}
        grads = {}
        grads_com = {}
        grads_m = {}
        scale = {}
        self.loss_coral_all = 0
        mul = 1   #mu = 100
        with autocast():
            if 'A' in self.modality:
                self.loss_coral_a = mul * coral(self.fa, self.fa_t)
                loss_data['A'] = self.loss_coral_a.item()
                self.loss_coral_all += self.loss_coral_a

            if 'V' in self.modality:
                self.loss_coral_v = mul * coral(self.fv, self.fv_t)
                loss_data['V'] = self.loss_coral_v.item()
                self.loss_coral_all += self.loss_coral_v

            if 'L' in self.modality:
                self.loss_coral_l = mul * coral(self.fl, self.fl_t)
                loss_data['L'] = self.loss_coral_l.item()
                self.loss_coral_all += self.loss_coral_l

            self.scaler.scale(self.loss_coral_all).backward(retain_graph=True)

            grads_all['all'] = []
            grads_all['all'].append(copy.deepcopy(self.feat_all_gradients))
            self.feat_all_gradients.zero_()

            #self.loss_coral_com = mul * coral(self.pred, self.pred_t)
            self.loss_coral_com = mul * coral(self.fall, self.fall_t)
            loss_data_m['A'] = loss_data_m['V'] =loss_data_m['L'] = self.loss_coral_com.item()
            self.scaler.scale(self.loss_coral_com).backward(retain_graph=True)

            grads_com['all'] = []
            grads_com['all'].append(copy.deepcopy(self.feat_all_gradients))
            self.feat_all_gradients.zero_()

        
        grads_all['all'] = torch.sum(grads_all['all'][0], dim=0)
        grads_com['all'] = torch.sum(grads_com['all'][0], dim=0)
        for t in self.modality:
            if t == 'A':
                grads['A'] = grads_all['all'][0: self.opt.embd_size_a]
                grads_all['all'] = grads_all['all'][self.opt.embd_size_a:]

                grads_m['A'] = grads_com['all'][0: self.opt.embd_size_a]
                grads_com['all'] = grads_com['all'][self.opt.embd_size_a:]
            
            if t == 'V':
                grads['V'] = grads_all['all'][0: self.opt.embd_size_v]
                grads_all['all'] = grads_all['all'][self.opt.embd_size_v:]

                grads_m['V'] = grads_com['all'][0: self.opt.embd_size_v]
                grads_com['all'] = grads_com['all'][self.opt.embd_size_v:]

            if t == 'L':
                grads['L'] = grads_all['all'][0: self.opt.embd_size_l]
                #grads_all['all'] = grads_all['all'][self.opt.embd_size_a:]
                grads_m['L'] = grads_com['all'][0: self.opt.embd_size_l]
            
        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, 'loss')  # 'l2'
        for t in self.modality: #'AVL':
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]
        
        gn_m = gradient_normalizers(grads_m, loss_data_m, 'loss')  # 'l2'
        for t in self.modality: #'AVL':
            for gr_i in range(len(grads_m[t])):
                grads_m[t][gr_i] = grads_m[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.

        # for t in 'AVL':
        #     grads[t] = grads[t][0]
        #     grads[t] = torch.sum(grads[t], dim=0)


        modality = self.modality #'AVL'
        nc = len(self.modality)
        M = nc
        #sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in modality])
        if self.use_cvx == 1:
            grad_all = torch.stack([grads[t] for t in modality]).cpu() #.to(self.device)
            b = torch.zeros(grad_all.shape[1]).cpu() #.to(self.device)
            sol = pte(grad_all, b, nc)

        if self.use_cvx_qp == 1:
            grad_all = torch.stack([grads[t] for t in modality])
            p = torch.matmul(grad_all, grad_all.T)
            p = p.cpu()
            sol = pte_qp(p, nc)
        
        if self.use_cfs == 1:
            pd = torch.tensor([torch.dot(grads[t], grads[t]) for t in modality])
            pd_inv = 1 / pd
            sol = pd_inv / sum(pd_inv)
        
        if self.use_full == 1:
            pd_1 = torch.tensor([torch.dot(grads[t], grads[t]) for t in modality])
            pd_2 = torch.tensor([torch.dot(grads_m[t], grads_m[t]) for t in modality])
            pd_x = torch.tensor([torch.dot(grads[t], grads_m[t]) for t in modality])
            diag = [i for i in pd_1]
            diag.append(sum(pd_2))
            diag = torch.tensor(diag)
            Q_t = diag
            Q = torch.diag(diag)
            Q[M, 0:M] = pd_x
            Q[0:M, M] = pd_x
            Q_t_inv = 1/Q_t
            sol = Q_t_inv / (sum(Q_t_inv))
        
        
        mod_all = self.modality+'J'
        for _, t in enumerate('AVLJ'):
            for i, item in enumerate(mod_all):
                if t == item:
                    scale[t] = float(sol[i])
        return scale


        
        
    def backward(self):
        """Calculate the loss for back propagation"""

        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CE_A = self.loss_CE_V = self.loss_CE_L = 0

        if 'A' in self.modality:
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
        if 'V' in self.modality:
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label)
        if 'L' in self.modality:
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)

    
        self.loss = self.loss_CE + self.weight * (self.loss_CE_A + self.loss_CE_V + self.loss_CE_L)

        self.loss.backward()


        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.5)

    def loss_cal(self, scale): 
        """Calculate the loss (for amp)"""
        #scale = self.weight_cal()
        scale_current = [0] * 4
        for i, t in enumerate('AVLJ'):
            if t in self.modality+'J':
                scale_current[i] = scale[t]
        scale_current = np.array(scale_current)
        beta = 0.2
        if np.isnan(scale_current).any():
            self.scale == self.scale
        else:
            self.scale = (1- beta) * self.scale + beta * scale_current
        
        scalea = {}
        scalea['A'], scalea['V'], scalea['L'], scalea['J'] = self.scale[0], self.scale[1], self.scale[2], self.scale[3]
        
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CE_t = self.criterion_ce(self.logits_t, self.label_t)
        if self.index_pl.shape[0] >=1:
            self.loss_CE_tpl = self.criterion_ce(self.logits_t[self.index_pl], self.pl)
        else:
            self.loss_CE_tpl = 0
        #self.loss_ee = entropy_re(self.feat)
        
        mul = 1e3 #1e5 #1e3
        mul2 = 1e-3
        self.loss_ee = mul2 * entropy_re(self.fall)
        self.loss_CE_A = self.loss_CE_V = self.loss_CE_L = 0
        self.loss_ee_A = self.loss_ee_V = self.loss_ee_L = 0
        self.loss_domdisc_a = self.loss_domdisc_v = self.loss_domdisc_l = 0
        self.loss_coral = self.loss_coral_a = self.loss_coral_v = self.loss_coral_l = 0
        self.loss_CEave_t = self.loss_CEave_ta = self.loss_CEave_tv = self.loss_CEave_tl = 0
        

        self.loss_domdisc_com = self.criterion_domdisc(self.domainCom, self.domain_combined_label)

        self.loss_coral = mul * coral(self.fall, self.fall_t)


        w = 0
        if 'A' in self.modality:
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
            self.loss_ee_A = mul2 * entropy_re(self.fa)
            self.loss_coral_a = mul * coral(self.fa, self.fa_t)

        if 'V' in self.modality:
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label)
            self.loss_ee_V = mul2 * entropy_re(self.fv)
            self.loss_coral_v = mul * coral(self.fv, self.fv_t)

        if 'L' in self.modality:
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)
            self.loss_ee_L = mul2 * entropy_re(self.fl)
            self.loss_coral_l = mul * coral(self.fl, self.fl_t)

        #self.loss_ada = scalea['J'] * self.loss_domdisc_com + scalea['A'] * self.loss_domdisc_a + scalea['V'] * self.loss_domdisc_v + scalea['L'] * self.loss_domdisc_l 
        self.loss_coralall = scalea['J'] * self.loss_coral + scalea['A'] * self.loss_coral_a + scalea['V'] * self.loss_coral_v + scalea['L'] * self.loss_coral_l

 
        self.loss = self.loss_CE + 0.5 * self.alpha * 0.1 * 10 * self.loss_CE_tpl + 0.1 * 20*0.05 * (self.loss_CE_A + self.loss_CE_V + self.loss_CE_L) + \
                0.1 * 10* 0.1 * self.loss_coralall + 0.001* 0.01 * (self.loss_ee + self.loss_ee_A + self.loss_ee_V + self.loss_ee_L)
            
        
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        if self.is_amp:
            self.optimizer.zero_grad()
            with autocast():
                self.forward()
                scale = self.weight_cal()  # Calculate weights after forward pass
                self.loss_cal(scale) 
            # backward
            self.optimizer.zero_grad() 
            self.scaler.scale(self.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # forward
            self.forward()   
            # backward

            self.optimizer.zero_grad() 
            self.backward()  
            self.optimizer.step()


