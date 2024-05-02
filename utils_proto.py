from __future__ import print_function
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime
import numpy as np
import copy
from scipy import stats
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import datasets, transforms, models
#from torch.hub import load_state_dict_from_url
#from torch.utils.model_zoo import load_url as load_state_dict_from_url
#from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, densenet121, DenseNet121_Weights, swin_t, Swin_T_Weights
#from torch.utils.tensorboard import SummaryWriter

#from models.wideresnet import *
#from models.resnet import *
#from models.densenet import *
#from models.resnext import *
#from models.simple import *
#from models.allconv import *
#from models.wideresnet import *
from loss_utils import *
from utils import *
from plot_methods import *

from sklearn.model_selection import train_test_split


def apply_orthoreg(model, lr, beta=0.001, lambd=10., epsilon=1e-6):
    """Loops through the layers of a CNN and applies orthoreg regularization.
       Apply it before "zero_grad()" or after "step()"
       Rodríguez, Pau, et al. "Regularizing cnns with locally constrained decorrelations." 
       ICLR (2017).

    Arguments:
        model {torch.nn.Module} -- network to regularize
        lr {float} -- current model learning rate

    Keyword Arguments:
        beta {float} -- regularization strength (default: {0.001})
        lambd {float} -- dampening (default: {10.})
        epsilon {[type]} -- numerical stability constant (default: {1e-6})
    """
    @torch.no_grad()
    def orthoreg(m):
        if type(m) == torch.nn.Conv2d:
            filters = m.weight.data.clone().view(m.weight.size(0), -1)
            norms = filters.norm(2, 1).view(-1, 1).expand_as(filters)
            filters.div_(norms + epsilon)
            grad = torch.mm(filters, filters.transpose(1, 0))
            grad = (grad * lambd) / (grad + np.exp(lambd))
            grad = grad * (1 - torch.eye(grad.size(0), dtype=grad.dtype, device=grad.device))
            grad = torch.mm(grad, filters)
            coeff = -1 * beta * lr
            m.weight.data.view(m.weight.size(0), -1).add_(grad * coeff)
    model.apply(orthoreg)

class Proximity(nn.Module):

    def __init__(self, device, num_classes=10, num_ftrs=512, k=0, protonorm=0, class_sched=[], ftr_sched=[]):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.num_ftrs = num_ftrs
        self.device = device
        #self.class_sched = torch.tensor(class_sched).to(device)  #as tensor?            
        #self.ftr_sched = torch.tensor(ftr_sched).to(device)      #as tensor?    
        #self.cent_per_class = cent_per_class                                                                                                                                              
        self.k = k
        self.magk = np.absolute(self.k)
        #self.decay_pow = decay_pow                                                                                                                                                             
        #self.decay_const = decay_const                                                                                                                                                                           
        self.largest = True if self.k > 0 else False
        #self.datanorm = datanorm                                                                                                                                                                                                    
        self.protonorm = protonorm
        #self.class_mask = torch.ones([self.num_classes],dtype=torch.bool, device=device)
        #self.class_mask[self.class_sched] = False

    def forward(self, x, labels, targets):
        
        batch_size = x.size(0)
        numT = targets.size(0)
        if self.protonorm:
            #targets_ = F.normalize(targets_)                                                                                                                                                                                         
            #need to be careful how you handle normalization if using a reduced cardinality                                                                                                                                         
            #should really be using the original magnitudes                                                                                                                                                                           
            x = F.normalize(x)

        if self.k != 0:
            #compute mask                                                                                                                                                                                                              
            with torch.no_grad():

                proto_0 = torch.zeros_like(targets)
                proto_1 = torch.zeros_like(targets)
                #following line grabs indices of top k values, and uses those indices to place or "scatter" ones from proto_1 onto proto_0                                                                                             
                idx_targets = torch.topk(targets, self.magk, dim=1, largest = self.largest)[1]
                #print (targets)                                                                                                                                                                                                      
                #print (idx_targets)                                                                                                                                                                                                   
                #for c, cl in enumerate(self.class_sched):
                    #idx_targets[cl][ftr_sched[n]:] = idx_targets[cl][0].item()                                                                                                                                                        
                    #idx_targets[cl][self.ftr_sched[c]:].fill_(idx_targets[cl][0].item())

                mask_targets = proto_0.scatter_(1, idx_targets, proto_1)

                #print (mask_targets[!=self.class_sched].shape)                                                                                                                                                                       
                #nullify classes                                                                                                                                                                                                      
                #mask_targets[!=self.class_sched] = 0.0*mask_targets[!=self.class_sched]                                                                                                                                              
                #mask_targets[self.class_mask] = 0.0
                #print (mask_targets)                                                                                                                                                                                                 
                mask_x = mask_targets[labels]

            targets_ = targets*mask_targets
            x_ = x*mask_x
        else:
            targets_ = targets
            x_ = x

        #x is [batch_size, num_ftr]
        #targets is [nclass, num_ftr]

        #res = torch.abs(x_ - targets_[labels])
        #loss = res.mean()
        #return loss
        
        distmat = torch.pow(x_, 2).sum(dim=1, keepdim=True).expand(batch_size, numT) + \
                  torch.pow(targets_, 2).sum(dim=1, keepdim=True).expand(numT, batch_size).t()
        distmat.addmm_(x_, targets_.t(), beta=1, alpha=-2)
        distmat.clamp_(1e-6,1e6)
                                                                                                                                                                               
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))   #[batch_size, num_classes]                                                                                                                                    
        res = mask*distmat
        #res = torch.max(res, dim=1)[0]                                                                                                                                                                                               
        #if self.k==0:                                                                                                                                                                                                                
        #    res /= self.num_ftrs                                                                                                                                                                                                    
        #else:                                                                                                                                                                                                                        
        #    res /= self.k
        
        loss = res.mean()

        return loss


class ProtoCovLoss(nn.Module):

    def __init__(self, device, num_classes=10, num_ftrs=512, k=0, protonorm=0, class_sched=[], ftr_sched=[]):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.num_ftrs = num_ftrs
        self.device = device
        #self.class_sched = torch.tensor(class_sched).to(device)  #as tensor?            
        #self.ftr_sched = torch.tensor(ftr_sched).to(device)      #as tensor?    
        #self.cent_per_class = cent_per_class                                                                                                                                              
        self.k = k
        self.magk = np.absolute(self.k)
        #self.decay_pow = decay_pow                                                                                                                                                             
        #self.decay_const = decay_const                                                                                                                                                                           
        self.largest = True if self.k > 0 else False
        #self.datanorm = datanorm                                                                                                                                                                                                    
        self.protonorm = protonorm
        #self.class_mask = torch.ones([self.num_classes],dtype=torch.bool, device=device)
        #self.class_mask[self.class_sched] = False

    def forward(self, x, labels, proto_latent):
        
        batch_size = x.size(0)
        numT = targets.size(0)
        if self.protonorm:
            #targets_ = F.normalize(targets_)                                                                                                                                                                                         
            #need to be careful how you handle normalization if using a reduced cardinality                                                                                                                                         
            #should really be using the original magnitudes                                                                                                                                                                           
            x = F.normalize(x)

        if self.k != 0:
            #compute mask                                                                                                                                                                                                              
            with torch.no_grad():

                proto_0 = torch.zeros_like(targets)
                proto_1 = torch.zeros_like(targets)
                #following line grabs indices of top k values, and uses those indices to place or "scatter" ones from proto_1 onto proto_0                                                                                             
                idx_targets = torch.topk(targets, self.magk, dim=1, largest = self.largest)[1]
                #print (targets)                                                                                                                                                                                                      
                #print (idx_targets)                                                                                                                                                                                                   
                #for c, cl in enumerate(self.class_sched):
                    #idx_targets[cl][ftr_sched[n]:] = idx_targets[cl][0].item()                                                                                                                                                        
                    #idx_targets[cl][self.ftr_sched[c]:].fill_(idx_targets[cl][0].item())

                mask_targets = proto_0.scatter_(1, idx_targets, proto_1)

                #print (mask_targets[!=self.class_sched].shape)                                                                                                                                                                       
                #nullify classes                                                                                                                                                                                                      
                #mask_targets[!=self.class_sched] = 0.0*mask_targets[!=self.class_sched]                                                                                                                                              
                #mask_targets[self.class_mask] = 0.0
                #print (mask_targets)                                                                                                                                                                                                 
                mask_x = mask_targets[labels]

            targets_ = targets*mask_targets
            x_ = x*mask_x
        else:
            targets_ = targets
            x_ = x


        distmat = torch.pow(x_, 2).sum(dim=1, keepdim=True).expand(batch_size, numT) + \
                  torch.pow(targets_, 2).sum(dim=1, keepdim=True).expand(numT, batch_size).t()
        distmat.addmm_(x_, targets_.t(), beta=1, alpha=-2)
        distmat.clamp_(1e-6,1e6)
        #distmat = torch.sqrt(distmat)     # [batch_size, numT]                                                                                                                                                                       
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))   #[batch_size, num_classes]                                                                                                                                    
        res = mask*distmat
        #res = torch.max(res, dim=1)[0]                                                                                                                                                                                               
        #if self.k==0:
        return 0


def calc_ftr_proto_data(args, model, device, nclass, ftr_length, loader, transformDict={},**kwargs):

    prototype_vectors0 = torch.zeros( [nclass, ftr_length//2], dtype=torch.float, device=device)
    prototype_vectors1 = torch.zeros( [nclass,ftr_length], dtype=torch.float, device=device)

    #prox_criterion = Proximity(device, num_classes=len(prototypes),k=args.k,protonorm=args.protonorm)

    pnts_per_class = float(len(loader.dataset) // nclass)

    print (pnts_per_class)


    for batch_idx, (data,y) in enumerate(loader):
        
        data, target = data.to(device), y.to(device)

        data_norm = transformDict['norm'](data)


        proto0, proto1, logits = model(data_norm)
            

        with torch.no_grad():
            for idx0, vec0 in enumerate(proto0):
                prototype_vectors0[y[idx0]] += vec0.clone()
            for idx1, vec1 in enumerate(proto1):
                prototype_vectors1[y[idx1]] += vec1.clone()

    prototype_vectors0.div_(pnts_per_class)
    prototype_vectors1.div_(pnts_per_class)

    print (torch.min(prototype_vectors1))
    print (torch.max(prototype_vectors1))

    prototype_vectors0.clamp_(0.0, 1.e6)
    prototype_vectors1.clamp_(0.0, 1.e6)

    #return unsorted, on cpu
    return prototype_vectors0, prototype_vectors1






def train_image_data(args, model, device, par_images, loader, iterations=20, mask=0, transformDict={},targets=-1,**kwargs):

    #def train_image_nodata(args, model, device, par_images, iterations=100, mask=0, transformDict={},targets=-1,**kwargs):

    #print ('Training images against one hot encodings')
    #(self, device, num_classes=10, cent_per_class=1, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):
    # criterion_prox = Proximity(device=device, 
    #     num_classes=kwargs['num_classes'], 
    #     cent_per_class=args.class_centers, 
    #     k=kwargs['k'],
    #     protonorm=kwargs['proto_norm'],
    #     psi=kwargs['psi'],
    #     kprox=kwargs['kprox'])
    #class Proximity(nn.Module):
    model.eval()
    image_lr = abs(args.image_step)
    #targets=targets.to(device)
    first_loss = 0.0


    #def __init__(self, device, num_classes=10, cent_per_class=1, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):



    criterion_prox = Proximity(device,
                                num_classes=len(par_images),
                                num_ftrs=args.ftrcnt,
                                k=0,
                                protonorm=args.proto_norm)

    #targets = label
    #print (targets)
    for i in range(iterations):
        for batch_idx, (data,y) in enumerate(loader):
        
            data, target = data.to(device), y.to(device)

            _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)

            #print (transformDict['norm'])
            data_norm = transformDict['norm'](data)
            _par_images_opt_norm = transformDict['norm'](_par_images_opt)
            #_par_images_opt_norm = transformDict['norm'](_par_images_opt.clone().detach())
            #_par_images_opt.data = _par_images_opt_norm.data

            L2_inp, logits = model(data_norm)
            L2_img, logits_img = model(_par_images_opt_norm)

            loss = criterion_prox(L2_inp, target, L2_img)
            #print (loss)

            #loss.backward(gradient=torch.ones_like(loss))
            loss.backward()


            with torch.no_grad():

                #if args.image_step > 0.0:
                #    image_gradients = args.image_step*torch.sign(_par_images_opt.grad)
                #else:
                #    image_gradients = args.par_grad_mult*_par_images_opt.grad
                #    image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
                    
                if batch_idx==0:
                    first_loss = loss.clone().cpu()
                if args.image_step > 0.0:
                    image_gradients = args.image_step*torch.sign(_par_images_opt.grad)
                elif args.image_step == 0.0:
                    image_gradients = args.par_grad_mult*_par_images_opt.grad
                    image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
                elif args.image_step < 0.0:
                    gradients_unscaled = _par_images_opt.grad
                    grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
                    image_gradients = image_lr*gradients_unscaled  / (grad_mag.view(-1, 1, 1, 1) + 1.e-6)

                #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
                #print (torch.mean(image_gradients))
                #print ("image gradients are ", image_gradients)
                par_images.add_(-image_gradients)
                # if roundDec >-1 :
                #     par_images = torch.round(par_images, decimals=roundDec)

                par_images.clamp_(0.0,1.0)

                if mask:
                    par_images = mask*par_images

                _par_images_opt.grad.zero_()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {}\t BatchID: {}\t Loss {:.6f}'.format(i, batch_idx, torch.mean(loss).item()))


    #with torch.no_grad():
    #    _par_images_final = par_images.clone().detach().requires_grad_(False).to(device)
    #    _par_images_final_norm = transformDict['norm'](_par_images_final)
    #    L2_img, logits_img = model(_par_images_final_norm)
    #    pred = logits_img.max(1, keepdim=True)[1]

        # print progress
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))

    #    if batch_idx % args.log_interval == 0:
    #        print('Train Epoch: {}\t BatchID: {}\t Loss {:.6f}'.format(epoch, batch_idx, torch.mean(loss).item()))

    return loss, par_images



def train_image_nodata(args, model, device, par_images, iterations=200, mask=0, transformDict={},targets=-1,**kwargs):


    #print (targets)
    model.multi_out=1
    model.eval()

    image_lr = abs(args.image_step)
    targets=targets.to(device)
    first_loss = 0.0
    

    for batch_idx in range(iterations):
    #for batch_idx, (data, target) in enumerate(par_images):
        #data, target = data.to(device), target.to(device)

        #data = data.to(device)
        #target = target.to(device)

        #print (data.shape)
        #print (target.shape)

        _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        #_par_images_opt = data.clone().detach().requires_grad_(True).to(device)

        #print (transformDict['norm'])
        #data = transformDict['norm'](data)
        _par_images_opt_norm = transformDict['norm'](_par_images_opt)
        #_par_images_opt_norm = transformDict['norm'](_par_images_opt.clone().detach())
        #_par_images_opt.data = _par_images_opt_norm.data

        #L2_inp, logits = model(data)
        L2_img, logits_img = model(_par_images_opt_norm)

        loss = F.cross_entropy(logits_img, targets, reduction='none')

        #loss = args.beta*criterion_prox(L2_inp, target, L2_img)

        loss.backward(gradient=torch.ones_like(loss))


        with torch.no_grad():
            if batch_idx==0:
                first_loss = loss.clone().cpu()
            if args.image_step > 0.0:
                image_gradients = args.image_step*torch.sign(_par_images_opt.grad)
            elif args.image_step == 0.0:
                image_gradients = args.par_grad_mult*_par_images_opt.grad
                image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
            elif args.image_step < 0.0:
                gradients_unscaled = _par_images_opt.grad
                grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
                image_gradients = image_lr*gradients_unscaled  / (grad_mag.view(-1, 1, 1, 1) + 1.e-6)  
            

            #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
            #print (torch.mean(image_gradients))
            #print ("image gradients are ", image_gradients)
            if (torch.mean(loss)>1e-6):
                par_images.add_(-image_gradients)
                #data.add_(-image_gradients)
                #if roundDec > -1:
                #    par_images = torch.round(par_images, decimals=roundDec)
                par_images.clamp_(0.0,1.0)
                #data.clamp_(0.0,1.0)

                if mask:
                    par_images = mask*par_images

                _par_images_opt.grad.zero_()
            else:
                par_images.clamp_(0.0,1.0)
                _par_images_opt.grad.zero_()
                break

        # print progress
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))

        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {}\t BatchID: {}\t Loss {:.6f}'.format(epoch, batch_idx, torch.mean(loss).item()))

    # with torch.no_grad():
    #     _par_images_final = par_images.clone().detach().requires_grad_(False).to(device)
    #     _par_images_final_norm = transformDict['norm'](_par_images_final)
    #     L2_img, logits_img = model(_par_images_final_norm)
    #     pred = logits_img.max(1, keepdim=True)[1]
    #     probs = F.softmax(logits_img)
    #     print (torch.max(logits_img,dim=1)[1])

    #model.multi_out=0
    #_par_images_opt = _par_images_opt.cpu()
    #_par_images_opt = 0.0
    #par_images = par_images.cpu()

    #return loss
    return loss

def all_pairs_L2(vectors):
    vectors = vectors.cpu()
    length = len(vectors)
    par_tens_flat = vectors.view(length,-1).cpu()

    distmatpow = torch.pow(par_tens_flat, 2).sum(dim=1, keepdim=True).expand(par_tens_flat.shape[0], par_tens_flat.shape[0]) + \
    torch.pow(par_tens_flat, 2).sum(dim=1, keepdim=True).expand(par_tens_flat.shape[0], par_tens_flat.shape[0]).t()

    distmat = torch.nan_to_num(torch.sqrt(distmatpow.addmm_(par_tens_flat, par_tens_flat.t(), beta=1, alpha=-2)))

    #return n(n-1) element matrix
    return distmat


def calc_data_proto_CS_covs(args, model, device, ftrpct_list, nclass, ftr_length, loader, HW=32,transformDict={}):


    prototype_vectors1 = torch.zeros([nclass,ftr_length], dtype=torch.float)
    model.eval()

    #prox_criterion = Proximity(device, num_classes=len(prototypes),k=args.k,protonorm=args.protonorm)

    pnts_per_class = float(len(loader.dataset) // nclass)

    print (pnts_per_class)

    diag_cov_list = [] 
    offdiag_cov_list = [] 
    min_offdiag_list = [] 
    mean_offdiag_list = [] 
    max_offdiag_list = [] 
    mean_var_list = [] 
    max_var_list = [] 

    diag_cov_unit_list = []
    offdiag_cov_unit_list = []
    
    min_offdiag_unit_list = []
    mean_offdiag_unit_list = []
    mean_offdiag_abs_unit_list = []
    max_offdiag_unit_list = []
    mean_var_unit_list = []
    max_var_unit_list = []
    #risk_covunit_cs = []
    risk_numer = []
    risk_numer_abs = []
    risk_denom = []

    risk_cheb = []
    risk_cantelli = []
    risk_cheb_abs = []
    risk_cantelli_abs = []


    corr_cur_unit_list = []
    diag_corr_unit_list = []
    offdiag_corr_unit_list = []
    min_offdiag_corr_unit_list = []
    mean_offdiag_corr_unit_list = []
    max_offdiag_corr_unit_list = []    



    ftr_vectors_by_class = []
    ftr_unit_vectors_by_class = []



    # for f, ft in enumerate(ftrpct_list):
    #     min_offdiag_unit_list.append([])
    #     mean_offdiag_unit_list.append([])
    #     max_offdiag_unit_list.append([])
    #     mean_var_unit_list.append([])
    #     max_var_unit_list.append([])
    #     risk_covunit_cs_by_class.append([])
    #     risk_numer_by_class.append([])
    #     risk_denom_by_class.append([])


    for _ in range(nclass):
        ftr_vectors_by_class.append([])
        ftr_unit_vectors_by_class.append([])

    model.multi_out=1

    for batch_idx, (data,y) in enumerate(loader):
        
        data, target = data.to(device), y.to(device)

        data_norm = transformDict['norm'](data)


        proto1, logits = model(data_norm)

        proto1 = proto1.cpu()
        proto1_unit = F.normalize(proto1, dim=1)
            

        with torch.no_grad():

            for idx1, vec1 in enumerate(proto1):
                prototype_vectors1[y[idx1]] += vec1.clone()

                ftr_vectors_by_class[y[idx1]].append(vec1.clone())

            for idx2, vec2 in enumerate(proto1_unit):

                ftr_unit_vectors_by_class[y[idx2]].append(vec2.clone())

    #potential small error here if stratification isnt perfect
    prototype_vectors1.div_(pnts_per_class)
    prototype_vectors1 = prototype_vectors1.cpu()
    prototypes_unit = F.normalize(prototype_vectors1, dim=1)

    print ("proto unit shape", prototypes_unit.shape)
    print ("proto unit 0", prototypes_unit[0])
    print ("proto unit 1", prototypes_unit[1])
    

    prototype_vectors_sort, proto_sort_idx = torch.sort(prototype_vectors1, dim=1)

    cos_mat_latent = prototypes_unit @ prototypes_unit.t()
    print (cos_mat_latent)
    cos_mat_latent_nondiag = cos_mat_latent.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1)
    L2_mat_latent = all_pairs_L2(prototype_vectors1)

    print ("cos mat latent shape", cos_mat_latent.shape)
    print ("cos_mat_latent_nondiag shape", cos_mat_latent_nondiag.shape)
    print ("cos_mat_latent non diag 0", cos_mat_latent_nondiag[0])


    for f, fp in enumerate(ftrpct_list):
        min_offdiag_unit_list.append([])
        mean_offdiag_unit_list.append([])
        mean_offdiag_abs_unit_list.append([])
        max_offdiag_unit_list.append([])
        mean_var_unit_list.append([])
        max_var_unit_list.append([])
        #risk_covunit_cs.append([])
        risk_numer.append([])
        risk_numer_abs.append([])
        risk_denom.append([])

        risk_cheb.append([])
        risk_cantelli.append([])
        risk_cheb_abs.append([])
        risk_cantelli_abs.append([])
        

        fp_cur = max(2, int(fp*ftr_length))
        fp_cur_corr = max(2, int(min(0.35,fp)*ftr_length))

        for nid in range(nclass):

            # #covariance
            # cur_stack = torch.stack(ftr_vectors_by_class[nid],dim=0)    #obs,ftrs
            # cur_stack_t = torch.transpose(cur_stack,0,1)   #ftrs,obs
            # cov_cur = torch.cov(cur_stack_t)
            
            # diag_cov = torch.diagonal(cov_cur,  offset=0)
            # offdiag_cov = torch.triu(cov_cur, diagonal=1)
            # min_offdiag = torch.min(offdiag_cov)
            # mean_offdiag = torch.mean(offdiag_cov)
            # max_offdiag = torch.max(offdiag_cov)
            # mean_var = torch.mean(diag_cov)
            # max_var = torch.max(diag_cov)

            #unit covariance
            
            cur_stack_unit = torch.stack(ftr_unit_vectors_by_class[nid],dim=0)    #obs,ftrs
            cur_stack_unit_t = torch.transpose(cur_stack_unit,0,1)   #ftrs,obs
            cur_proto_unit_t = prototypes_unit[nid].clone().unsqueeze(1).expand_as(cur_stack_unit_t)
            #cur_class_latents = pert_latent_t[k][sorted_latent_indices[k]].clone()
            cur_proto_unit_t_sort = cur_proto_unit_t[proto_sort_idx[nid]]
            cur_stack_unit_t_sort = cur_stack_unit_t[proto_sort_idx[nid]]
            #if nid==0:
            #    print (cur_stack_unit_t_sort)
            random_var = cur_proto_unit_t_sort*cur_stack_unit_t_sort
            #cur_features = cur_stack_unit_t_sort[int(ftr_length-fp_cur):]
            cur_features = random_var[int(ftr_length-fp_cur):]
            #print (cur_stack_unit_t_sort.shape)
            #if nid==0:
            #    print (cur_features)
            cov_cur_unit = torch.cov(cur_features)


            total_ftrs = len(cov_cur_unit)
            #ftrs_to_count= max(2,int(0.35*total_ftrs))

            
            diag_cov_unit = torch.diagonal(cov_cur_unit,  offset=0)
            offdiag_cov_unit = torch.triu(cov_cur_unit, diagonal=1)
            cov_unit_triu = torch.triu(cov_cur_unit)
            #diag_cov_unit_sort, dcus_id = torch.sort(diag_cov_unit)
            #offdiag_cov_unit_sort, odcus_id = torch.sort(offdiag_cov_unit)

            #top_contributions = torch.mean(torch.sort(cov_cur_unit_triu.view(-1))[total_ftrs-ftrs_to_count:])
            numer = torch.sum(cov_cur_unit) # / (0.5*total_ftrs + 0.5*(total_ftrs**2))     #(n^2 - n)/2  + n      n^2/2 + n/2
            numer_abs = torch.sum(torch.abs(cov_cur_unit.clone())) # / (0.5*total_ftrs + 0.5*(total_ftrs**2))
            
            risk_numer[f].append(numer.clone())
            risk_numer_abs[f].append(numer_abs.clone())
            
            dissim_vector = (1.0-cos_mat_latent_nondiag[nid])**2.0
            #print ("dissim vector shape", dissim_vector.shape)
            #print ("dissim vector", dissim_vector)
            
            #print (dissim_vector.shape)
            #print (dissim_vector)
            denom = torch.mean(dissim_vector.clone())
            #sub_risk = 0.0

            #for val in dissim_vector:
            #    sub_risk += numer / val

            risk_denom[f].append(denom.clone())
            
            #risk_covunit_cs[f].append(sub_risk.clone())
            risk_cheb[f].append((numer/denom).clone())
            risk_cheb_abs[f].append((numer_abs/denom).clone())
            risk_cantelli[f].append((numer / (numer+denom)).clone())
            risk_cantelli_abs[f].append((numer_abs/ (numer_abs + denom)).clone())


            min_offdiag_unit = torch.min(offdiag_cov_unit)
            mean_offdiag_unit = torch.mean(offdiag_cov_unit)
            mean_offdiag_abs_unit = torch.mean(torch.abs(offdiag_cov_unit.clone()))
            max_offdiag_unit = torch.max(offdiag_cov_unit)
            mean_var_unit = torch.mean(diag_cov_unit)
            max_var_unit = torch.max(diag_cov_unit)

            # #unit correlation
            # corr_cur_unit = torch.corrcoef(cur_stack_unit_t)
            # diag_corr_unit = torch.diagonal(corr_cur_unit,  offset=0)
            # offdiag_corr_unit = torch.triu(corr_cur_unit, diagonal=1)
            # min_offdiag_corr_unit = torch.min(offdiag_corr_unit)
            # mean_offdiag_corr_unit = torch.mean(offdiag_corr_unit)
            # max_offdiag_corr_unit = torch.max(offdiag_corr_unit)

            # diag_cov_list.append(diag_cov.clone())
            # offdiag_cov_list.append(offdiag_cov.clone())
            # min_offdiag_list.append(min_offdiag.clone()) 
            # mean_offdiag_list.append(mean_offdiag.clone())
            # max_offdiag_list.append(max_offdiag.clone())
            # mean_var_list.append(mean_var.clone()) 
            # max_var_list.append(max_var.clone()) 

            #diag_cov_unit_list.append(diag_cov_unit.clone())
            #offdiag_cov_unit_list.append(offdiag_cov_unit.clone())

            min_offdiag_unit_list[f].append(min_offdiag_unit.clone())
            mean_offdiag_unit_list[f].append(mean_offdiag_unit.clone())
            mean_offdiag_abs_unit_list[f].append(mean_offdiag_abs_unit.clone())
            max_offdiag_unit_list[f].append(max_offdiag_unit.clone())
            mean_var_unit_list[f].append(mean_var_unit.clone())
            max_var_unit_list[f].append(max_var_unit.clone())


            # corr_cur_unit_list.append(corr_cur_unit.clone())
            # diag_corr_unit_list.append(diag_corr_unit.clone())
            # offdiag_corr_unit_list.append(offdiag_corr_unit.clone())
            # min_offdiag_corr_unit_list.append(min_offdiag_corr_unit.clone())
            # mean_offdiag_corr_unit_list.append(mean_offdiag_corr_unit.clone())
            # max_offdiag_corr_unit_list.append(max_offdiag_corr_unit.clone())


    #diag_cov_list = torch.mean(torch.stack(diag_cov_list,dim=0))
    #offdiag_cov_list = torch.mean(torch.stack(offdiag_cov_list,dim=0))
    
    # min_offdiag_list= torch.sum(torch.stack(min_offdiag_list,dim=0)) 
    # mean_offdiag_list= torch.sum(torch.stack(mean_offdiag_list,dim=0))
    # max_offdiag_list= torch.sum(torch.stack(max_offdiag_list,dim=0))
    # mean_var_list= torch.sum(torch.stack(mean_var_list,dim=0)) 
    # max_var_list= torch.sum(torch.stack(max_var_list,dim=0)) 

    #diag_cov_unit_list= torch.mean(torch.stack(diag_cov_unit_list,dim=0))
    #offdiag_cov_unit_list= torch.mean(torch.stack(offdiag_cov_unit_list,dim=0))
    
    #min_offdiag_unit_list= torch.sum(torch.stack(min_offdiag_unit_list,dim=0))
    #mean_offdiag_unit_list= torch.sum(torch.stack(mean_offdiag_unit_list,dim=0))
    #max_offdiag_unit_list= torch.sum(torch.stack(max_offdiag_unit_list,dim=0))
    #mean_var_unit_list= torch.sum(torch.stack(mean_var_unit_list,dim=0))
    #max_var_unit_list= torch.sum(torch.stack(max_var_unit_list,dim=0))


    #corr_cur_unit_list= torch.mean(torch.stack(corr_cur_unit_list,dim=0))
    #diag_corr_unit_list= torch.mean(torch.stack(diag_corr_unit_list,dim=0))
    #offdiag_corr_unit_list= torch.mean(torch.stack(offdiag_corr_unit_list,dim=0))
    
    # min_offdiag_corr_unit_list= torch.mean(torch.stack(min_offdiag_corr_unit_list,dim=0))
    # mean_offdiag_corr_unit_list= torch.mean(torch.stack(mean_offdiag_corr_unit_list,dim=0))
    # max_offdiag_corr_unit_list= torch.mean(torch.stack(max_offdiag_corr_unit_list,dim=0))

    #risk_numer = torch.sum(torch.stack(risk_numer_by_class, dim=0))   #risk_numer_by_class.append(numer.clone())
    #risk_denom = torch.sum(torch.stack(risk_denom_by_class, dim=0))
    #risk_covunit_cs = torch.sum(torch.stack(risk_covunit_cs_by_class,dim=0))

    datalist_return = []

    datalist_return.append(min_offdiag_unit_list)
    datalist_return.append(mean_offdiag_unit_list)
    datalist_return.append(max_offdiag_unit_list)
    datalist_return.append(mean_var_unit_list)
    datalist_return.append(max_var_unit_list)
    datalist_return.append(mean_offdiag_abs_unit_list)



    #get prototype images corresponding to prototype_vectors1
    #par_images_random = torch.rand([nclass,3,HW,HW],dtype=torch.float, device=device)
    proto_images = torch.rand([nclass,3,HW,HW],dtype=torch.float)
    #last_loss, proto_images = train_image_data(args, model, device, par_images_random, loader, iterations=5, mask=0, transformDict=transformDict)
    #proto_images = proto_images.cpu()



    #return unsorted, on cpu
    return prototype_vectors1, proto_images.clone(), cos_mat_latent, L2_mat_latent, risk_numer, risk_numer_abs, risk_denom, risk_cheb, risk_cheb_abs, risk_cantelli, risk_cantelli_abs, datalist_return


def estimate_chebyshev(args, model, device, ftrpct_list, prototype_inputs, cos_mat_latent, l2_mat_latent, transformDict, nclass,HW=32):

    print ("computing perturbations")
    perturbed_latents = []
    perturbed_latents_unit = []
    model.multi_out = 1

    with torch.no_grad():
        p_fwd = prototype_inputs.clone().to(device)
    
        p_fwd_norm = transformDict['norm'](p_fwd)

        latent_baseline, logits_baseline = model(p_fwd_norm)

        sorted_latent, sorted_latent_indices = torch.sort(latent_baseline,dim=1)            #nclass,numftr
        sorted_latent_indices = sorted_latent_indices.cpu()


    #all_inds = np.arange(nclass)
    p_magnitude=0.03

    for itx in range(12):

        with torch.no_grad():

            p_fwd = prototype_inputs.clone().to(device)
            perturbation = torch.rand([nclass,3,HW,HW],dtype=torch.float, device=device)

            #class centers for "same label" start close together
            p_fwd += p_magnitude*perturbation
            p_fwd.clamp_(0.0,1.0)

        
            p_fwd_norm = transformDict['norm'](p_fwd)

            latent_pert, logits_pert = model(p_fwd_norm)

            latent_pert_unit = F.normalize(latent_pert,dim=1)

            perturbed_latents.append(latent_pert.clone().cpu())
            perturbed_latents_unit.append(latent_pert_unit.clone().cpu())

    perturbed_latents = torch.stack(perturbed_latents, dim=1)   #[nclass, num_samples, 512]
    perturbed_latents_unit = torch.stack(perturbed_latents_unit, dim=1)

    pert_latent_t = torch.transpose(perturbed_latents,1,2)      #[nclass, ftr vars, obversations]
    pert_latent_unit_t = torch.transpose(perturbed_latents_unit,1,2)

    class_covs = []
    class_covs_unit = []
    class_corrs = []
    class_covs_abs = []
    class_covs_unit_abs = []

    class_min_offdiag_unit = []
    class_mean_offdiag_unit = []
    class_max_offdiag_unit = []
    class_mean_var_unit = []
    class_max_var_unit = []
    class_mean_abs_offdiag_unit = []

    dis_sim_matrix = ((1.0 - cos_mat_latent)**2.0).masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).clone()
    dis_l2_matrix = (l2_mat_latent**2.0).masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).clone()  

    #print ("dis sim matrix shape ", dis_sim_matrix.shape)

    #unified_Mg_covcov = []
    #unified_Mg_rhorho = []

    cur_ftrs = latent_pert.shape[1]

    for f, fp in enumerate(ftrpct_list):

        class_covs.append([])
        class_covs_unit.append([])
        class_corrs.append([])
        class_covs_abs.append([])
        class_covs_unit_abs.append([])

        class_min_offdiag_unit.append([])
        class_mean_offdiag_unit.append([])
        class_max_offdiag_unit.append([])
        class_mean_var_unit.append([])
        class_max_var_unit.append([])
        class_mean_abs_offdiag_unit.append([])

        fp_cur = max(2, int(fp*cur_ftrs))
        fp_cur_corr = max(2, int(min(0.35,fp)*cur_ftrs))

        for k in range(nclass):
            cur_class_latents = pert_latent_t[k][sorted_latent_indices[k]].clone()  #sorted by original protos indices
            cur_class_latent_obs = cur_class_latents[int(cur_ftrs - fp_cur):]
            cur_class_latent_obs_corr = cur_class_latents[int(cur_ftrs - fp_cur_corr):]

            #cur_class_latents_unit = pert_latent_unit_t[int(cur_ftrs - fp_cur):]
            #cur_class_latents_unit_obs = cur_class_latents_unit[k][sorted_latent_indices[k]].clone()
            cur_class_latents_unit = pert_latent_unit_t[k][sorted_latent_indices[k]].clone()
            cur_class_latents_unit_obs = cur_class_latents_unit[int(cur_ftrs-fp_cur):]


            cur_cov = torch.cov(cur_class_latent_obs)            #[fp_cur,fp_cur]
            cur_cov_unit = torch.cov(cur_class_latents_unit_obs)    #[fp_cur, fp_cur]
            diag_cov_unit = torch.diagonal(cur_cov_unit,  offset=0)
            offdiag_cov_unit = torch.triu(cur_cov_unit, diagonal=1)
            #cov_unit_triu = torch.triu(cur_cov_unit)

            #extract same information as data-based prototype assessment
            min_offdiag_unit = torch.min(offdiag_cov_unit)
            mean_offdiag_unit = torch.mean(offdiag_cov_unit)
            max_offdiag_unit = torch.max(offdiag_cov_unit)
            mean_var_unit = torch.mean(diag_cov_unit)
            max_var_unit = torch.max(diag_cov_unit)
            mean_offdiag_abs_unit = torch.mean(torch.abs(offdiag_cov_unit.clone()))

            class_min_offdiag_unit[f].append(min_offdiag_unit.clone())
            class_mean_offdiag_unit[f].append(mean_offdiag_unit.clone())
            class_max_offdiag_unit[f].append(max_offdiag_unit.clone())
            class_mean_var_unit[f].append(mean_var_unit.clone())
            class_max_var_unit[f].append(max_var_unit.clone())
            class_mean_abs_offdiag_unit[f].append(mean_offdiag_abs_unit.clone())


            cur_corr = torch.corrcoef(cur_class_latent_obs_corr)      #[fp_cur,fp_cur]
            cur_cov_abs = torch.abs(cur_cov)
            cur_cov_unit_abs = torch.abs(cur_cov_unit)


            #class_covs[f].append(torch.mean(cur_cov.masked_select(~torch.eye(fp_cur, dtype=bool)).view(fp_cur,fp_cur-1)).cpu().clone())
            class_corrs[f].append(torch.mean(cur_corr.masked_select(~torch.eye(fp_cur_corr, dtype=bool)).view(fp_cur_corr,fp_cur_corr-1)).cpu().clone())
            #class_covs_unit[f].append(torch.mean(cur_cov_unit.masked_select(~torch.eye(fp_cur, dtype=bool)).view(fp_cur,fp_cur-1)).cpu().clone())
            #class_covs_abs[f].append(torch.mean(cur_cov_abs.masked_select(~torch.eye(fp_cur, dtype=bool)).view(fp_cur,fp_cur-1)).cpu().clone())
            #class_covs_unit_abs[f].append(torch.mean(torch.mean(cur_cov_unit_abs.masked_select(~torch.eye(fp_cur, dtype=bool)).view(fp_cur,fp_cur-1)).cpu().clone()))

            cur_cov_ftr = len(cur_cov)
            class_covs[f].append(torch.sum(cur_cov))
            #class_covs[f].append(torch.sum(torch.triu(cur_cov)) / (0.5*cur_cov_ftr + (0.5*cur_cov_ftr**2)) )
            #class_corrs[f].append(torch.sum(torch.triu(cur_corr)) / (0.5*cur_ftrs + 0.5*cur_ftrs**2) )
            class_covs_unit[f].append(torch.sum(cur_cov_unit))
            #class_covs_unit[f].append(torch.sum(torch.triu(cur_cov_unit)) / (0.5*cur_cov_ftr + (0.5*cur_cov_ftr**2)) )
            class_covs_abs[f].append(torch.sum(cur_cov_abs))
            #class_covs_abs[f].append(torch.sum(torch.triu(cur_cov_abs)) / (0.5*cur_cov_ftr + (0.5*cur_cov_ftr**2)) )
            class_covs_unit_abs[f].append(torch.sum(cur_cov_unit_abs))
            #class_covs_unit_abs[f].append(torch.sum(torch.triu(cur_cov_unit_abs)) / (0.5*cur_cov_ftr + (0.5*cur_cov_ftr**2)) )


    uni_Mg_cov_list = []
    simp_Mg_cov_list = []
    uni_Mg_corr_list = []
    simp_Mg_corr_list = []

    risk_cov_l2_list = []
    risk_l2_cov_list = []
    risk_covunit_cs_list = []
    risk_cs_covunit_list = []
    risk_covabs_l2_list = []
    risk_covunitabs_cs_list = []
            

    for f, fp in enumerate(ftrpct_list):

        cov_stack_raw = torch.stack(class_covs[f], dim=0).clone()
        cov_stack_unit_raw = torch.stack(class_covs_unit[f], dim=0).clone()
        cov_stack_abs_raw = torch.stack(class_covs_abs[f], dim=0).clone()
        cov_stack_unit_abs_raw = torch.stack(class_covs_unit_abs[f],dim=0).clone()

        cov_stack = torch.stack(class_covs[f], dim=0).clone()
        cov_stack[torch.logical_and(cov_stack>=0.0, cov_stack<=1.e-5)] = 1.e-5
        cov_stack[torch.logical_and(cov_stack<0.0, cov_stack>=-1.e-5)] = -1.e-5

        cov_stack_unit = torch.stack(class_covs_unit[f],dim=0).clone()
        cov_stack_unit[torch.logical_and(cov_stack_unit>=0.0, cov_stack_unit<=1.e-5)] = 1.e-5
        cov_stack_unit[torch.logical_and(cov_stack_unit<0.0, cov_stack_unit>=-1.e-5)] = -1.e-5
        #all_pairs_class_cov = (cov_stack @ cov_stack.t())
        cov_row_expand = cov_stack.unsqueeze(1).expand(-1, nclass)
        cov_row_unit_expand = cov_stack_unit.unsqueeze(1).expand(-1,nclass)
        cov_row_raw_expand = cov_stack_raw.unsqueeze(1).expand(-1, nclass)
        cov_row_unit_raw_expand = cov_stack_unit_raw.unsqueeze(1).expand(-1, nclass)
        #if f==0:
        #    print (cov_row_unit_raw_expand)
        cov_row_abs_expand = cov_stack_abs_raw.unsqueeze(1).expand(-1, nclass)
        cov_row_unit_abs_raw_expand = cov_stack_unit_abs_raw.unsqueeze(1).expand(-1,nclass)


        #print ("cov_row_expand ", cov_row_expand.shape)
        #uni_Mg_cov_list.append(torch.sum(dis_sim_matrix.div(cov_row_expand).masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1)).cpu().clone())
        simp_Mg_cov_list.append(torch.mean(cov_row_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_sim_matrix)).cpu().clone())
        #simple_Mg_covcov_hyper_list[f].append(torch.mean(dis_sim_matrix.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))/torch.mean(cov_row_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1)).cpu().clone())
        uni_Mg_cov_list.append(torch.mean(cov_row_unit_raw_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_sim_matrix)).cpu().clone())


        corr_stack = torch.stack(class_corrs[f], dim=0).clone()
        corr_stack[torch.logical_and(corr_stack>=0.0, corr_stack<=1.e-4)] = 1.e-4
        corr_stack[torch.logical_and(corr_stack<0.0, corr_stack>=-1.e-4)] = -1.e-4
        #all_pairs_class_corr = corr_stack @ corr_stack.t()
        corr_row_expand = corr_stack.unsqueeze(1).expand(-1, nclass)
        #print ("all pairs class corr mat shape ", all_pairs_class_corr)
        uni_Mg_corr_list.append(torch.mean(dis_sim_matrix.div(corr_row_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1)).cpu().clone()))
        simp_Mg_corr_list.append(torch.mean(dis_sim_matrix)/torch.mean(corr_row_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1)).cpu().clone())

        risk_cov_l2_list.append(torch.sum(cov_row_raw_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_l2_matrix)).cpu().clone())
        risk_l2_cov_list.append(torch.sum(dis_l2_matrix.div(cov_row_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))).cpu().clone())

        risk_covunit_cs_list.append(torch.sum(cov_row_unit_raw_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_sim_matrix)).cpu().clone())
        risk_cs_covunit_list.append(torch.sum(dis_sim_matrix.div(cov_row_unit_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1))).cpu().clone())

        risk_covabs_l2_list.append(torch.sum(cov_row_abs_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_l2_matrix)).cpu().clone())
        risk_covunitabs_cs_list.append(torch.sum(cov_row_unit_abs_raw_expand.masked_select(~torch.eye(nclass, dtype=bool)).view(nclass,nclass-1).div(dis_sim_matrix)).cpu().clone())

    outside_class_covs = []
    outside_class_corrs = []


    #between class covariances
    #comment out to save time
    #fp_cur = max(2, int(0.05*cur_ftrs))

    #for k1 in range(nclass):
    #    for k2 in range(nclass):
    #        if k2 > k1:
    #            cov_sum = torch.tensor(0.0)
    #            corr_sum = torch.tensor(0.0)
    #            count = 0.0
    #
    #            k1_class_latents = pert_latent_t[k1][sorted_latent_indices[k1]].clone()   #[sorted ftr, observations]
    #            k1_class_latent_obs = k1_class_latents[int(cur_ftrs - fp_cur):]
    #            #notice that we still sort by k1
    #            k2_class_latents = pert_latent_t[k1][sorted_latent_indices[k2]].clone()   #[look at observations around same prototype image in image space, but pick features important to other classes  
    #            k2_class_latent_obs = k2_class_latents[int(cur_ftrs - fp_cur):]
    #
    #            obs_cat = torch.cat((k1_class_latent_obs, k2_class_latent_obs),dim=0)
    #
    #            #print ("obs cat shape ", obs_cat.shape)
    #
    #            cur_outside_class_cov = torch.cov(obs_cat)
    #            cur_outside_class_corr = torch.corrcoef(obs_cat)
    #
    #            #print ("cur outside class cov shape", cur_outside_class_cov.shape)
    #
    #print ("fp_cur ", fp_cur)
    #            #print (" cur_ftrs ", cur_ftrs)
    #
    #            for i in range(fp_cur):
    #                for j in range(fp_cur, 2*fp_cur):
    #                    if torch.abs(cur_outside_class_corr[i,j]) > 0.05:
    #                        cov_sum += cur_outside_class_cov[i,j]
    #                        corr_sum += cur_outside_class_corr[i,j]
    #                        count += 1.0

    #            if count == 0.0:
    #                count = 1.0
                    
                    
    #            outside_class_covs.append(cov_sum / count)
    #            outside_class_corrs.append(corr_sum / count)
    
                


    dissim_off = torch.mean(dis_sim_matrix)
    disL2_off = torch.mean(dis_l2_matrix)

    #class_covs_unit[0] has a list of class sums of over their respective covariance matrices

    return [class_covs, 
            class_corrs, 
            class_covs_unit, 
            uni_Mg_cov_list, 
            simp_Mg_cov_list, 
            uni_Mg_corr_list, 
            simp_Mg_corr_list, 
            outside_class_covs, 
            outside_class_corrs, 
            dissim_off,
            disL2_off, 
            risk_cov_l2_list,
            risk_l2_cov_list,
            risk_covunit_cs_list,
            risk_cs_covunit_list,
            risk_covabs_l2_list,
            risk_covunitabs_cs_list,
            class_min_offdiag_unit,
            class_mean_offdiag_unit,
            class_max_offdiag_unit,
            class_mean_var_unit,
            class_max_var_unit,
            class_covs_unit_abs]
