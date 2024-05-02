import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.resnet import *
from torchvision import transforms

class kldivproto(nn.Module):

    def __init__(self, device,num_classes=10):
        super(kldivproto, self).__init__()
        self.num_classes = num_classes
        self.device = device




    def forward(self, p, labels, q):

        # x is [batch, 1024]
        # labels [10]
        # targets [10, 1024]
        #targets.t [1024, 10]
        # x @ targets.t  = [batch,10] (all possible dot products for each of observations)

        batch_size = q.size(0)
        #distmat is summation of all 1024 elements squared, expanded to [batch, 10]
        #print (torch.min(p))
        p = p +0.0001
        cont2 = q @ p.t()  #[batch, protos]
        cont1 = torch.sum(p*p.log(),dim=1)  #[protos], elementwise mult
        klsum = cont1.unsqueeze(0).expand(batch_size,-1) - cont2  #elementwise subtraction, #[batch, protos]
        
        

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        klsum = mask*klsum

        loss = torch.sum(klsum) / batch_size
        
        return loss

class Diversity(nn.Module):

    def __init__(self, num_classes=10, margin = 1.0):
        super(Diversity, self).__init__()
        self.num_classes = num_classes
        self.margin = margin


    def forward(self, x):

        batch_size = x.size(0)

        d = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        distmat = d + d.t()
        distmat.addmm_(x, x.t(), beta=1, alpha=-2)

        distmat = 1e-6 + torch.sqrt(1e-3 + distmat)
        distmat.fill_diagonal_(self.margin + 1.0)

        #classes = torch.arange(self.num_classes).long().to(DEVICE)

        #labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        #negate for conprox
        #mask = ~labels.eq(classes.expand(batch_size, self.num_classes))

        distmat = torch.where((self.margin - distmat) < 0.0, 0.0*distmat, self.margin - distmat)
        distmat = torch.triu(distmat)
        #distmat = (mask*distmat)**2

        loss = distmat.mean()
        return loss

class ProxCos(nn.Module):

    def __init__(self, device, k=0):
        super(ProxCos, self).__init__()
        self.k = k


    def forward(self, x):

        distmat = x @ x.t()
        distmat = torch.triu(distmat, diagonal=1)
        loss = distmat.mean()
        return loss

#parser.add_argument('--maxmean', default=1, type=int,
#                    help='if 1, will use topk maxes from each row, if 0, topk means from cossim matrix')
#parser.add_argument('--proxpwr', default=1.0, type=float,
#                    help='power of the L2 dist on data to prototype')
#parser.add_argument('--topkprox', default=0, type=int,
#                    help='if not 0, will select only topk maxes from kprox selection ie top10 of top5 maxes')
#parser.add_argument('--hsphere', default=0, type-int,
#                    help='shrink variance on magnitudes to speed convergence')

class WeightedProximity(nn.Module):

    def __init__(self, device, num_classes=10, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):
        super(WeightedProximity, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.k = k
        self.magk = np.absolute(self.k)
        #self.decay_pow = decay_pow
        #self.decay_const = decay_const

        self.largest = True if self.k > 0 else False
        #self.datanorm = datanorm
        self.protonorm = protonorm
        self.psi=psi
        self.kprox=kprox

        self.maxmean= maxmean
        self.proxpwr = proxpwr
        self.topkprox = topkprox
        self.hsphere = hsphere




    def forward(self, x, labels, targets, weights):

        # x is [batch, 1024]
        # labels [10]
        # targets [10, 1024]
        #targets.t [1024, 10]
        # x @ targets.t  = [batch,10] (all possible dot products for each of observations)

        batch_size = x.size(0)
        numT = targets.size(0)

        if self.k != 0:
            #compute mask
            with torch.no_grad():
                # x_0 = torch.zeros_like(x)
                # x_1 = torch.ones_like(x)
                # idx_x = torch.topk(x, np.absolute(self.k), dim=1, largest = np.sign(self.k))[1]
                # mask_x = x_0.scatter_(1, idx_x, x_1)

                proto_0 = torch.zeros_like(targets)
                proto_1 = torch.ones_like(targets)
                idx_targets = torch.topk(targets, self.magk, dim=1, largest = self.largest)[1]
                mask_targets = proto_0.scatter_(1, idx_targets, proto_1)

                mask_x = mask_targets[labels]

            targets_ = targets*mask_targets
            x_ = x*mask_x
        else:
            targets_ = targets
            x_ = x

        if self.protonorm:
            targets_ = F.normalize(targets_)
            x_ = F.normalize(x_)


        #distmat is summation of all 1024 elements squared, expanded to [batch, 10]
        distmat = torch.pow(x_, 2).sum(dim=1, keepdim=True).expand(batch_size, numT) + \
                  torch.pow(targets_, 2).sum(dim=1, keepdim=True).expand(numT, batch_size).t()
        distmat.addmm_(x_, targets_.t(), beta=1, alpha=-2)
        distmat.clamp_(1e-6,1e6)
        #distmat = torch.sqrt(distmat)     # [batch_size, numT]

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))   #[batch_size, num_classes]

        # if self.cent_per_class > 1:
        #     y = torch.chunk(distmat, self.cent_per_class, dim=1)    #should give a [batch_size, num_classes] chunk
        #     res = y[0]*mask
        
        #     for i in range(1,self.cent_per_class):
        #         res = torch.minimum(res,y[i]*mask)
        # else:
       
        res = mask*distmat

        #print (res.shape)
        #print (weights.shape)
        #print (weights)
        #print (res)

        #print (torch.max(res,dim=1)[0])
        

        res = weights*torch.max(res, dim=1)[0]

        if self.k==0:
            res /= 512
        else:
            res /= self.k

        loss = res.mean()

        #print (loss)

        #print (loss)

        if self.hsphere:
            mags = torch.linalg.norm(x_,dim=1)
            with torch.no_grad():
                mu = mags.mean()
            hloss = 0.5*((mags-mu.expand_as(mags))**2).mean()
            loss += hloss

        if self.psi <=0.0:
            return loss
        
        if self.protonorm:
            targets_norm = targets_
        else:
            targets_norm = F.normalize(targets_)

        #till 0129 am    
        #prox_sim  = targets_norm @ targets_norm.t()
        #prox_sim = (torch.triu(prox_sim, diagonal=1))**2

        #loss_sim = prox_sim.mean()

        #loss_total = (1/1.414)*loss + self.psi*loss_sim

        #startin 0129 pm
        prox_sim = targets_norm @ targets_norm.t()
        prox_sim.fill_diagonal_(0)
        prox_sim = prox_sim**2
        #maxes = torch.max(prox_sim, dim=1)[0]

        if self.maxmean:
            maxes = torch.topk(prox_sim,k=self.kprox,dim=1,sorted=False)[0]
            #print (maxes)
            if self.topkprox > 0:
                loss_sim = (torch.topk(maxes,k=self.topkprox,dim=0,sorted=False)[0]).mean()
            else:
                loss_sim = maxes.mean()
            #prox_sim_no_diag = (torch.triu(prox_sim, diagonal=1) + torch.tril(prox_sim, diagonal=-1))**2
        else:
            means = torch.mean(prox_sim,dim=1)
            loss_sim = (torch.topk(means,k=self.kprox,dim=0,sorted=False)[0]).mean()

        #print (loss_sim)
            
        loss += self.psi*loss_sim


        #if self.decay_pow > 0.0:
        #    loss -= self.decay_const*(torch.linalg.norm(targets_, 2, dim=1)**self.decay_pow).mean()

        return loss


class Proximity(nn.Module):

    def __init__(self, device, num_classes=10, cent_per_class=1, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.cent_per_class = cent_per_class
        self.k = k
        self.magk = np.absolute(self.k)
        #self.decay_pow = decay_pow
        #self.decay_const = decay_const

        self.largest = True if self.k > 0 else False
        #self.datanorm = datanorm
        self.protonorm = protonorm
        self.psi=psi
        self.kprox=kprox

        self.maxmean= maxmean
        self.proxpwr = proxpwr
        self.topkprox = topkprox
        self.hsphere = hsphere




    def forward(self, x, labels, targets):

        # x is [batch, 1024]
        # labels [10]
        # targets [10, 1024]
        #targets.t [1024, 10]
        # x @ targets.t  = [batch,10] (all possible dot products for each of observations)

        batch_size = x.size(0)
        numT = targets.size(0)

        if self.k != 0:
            #compute mask
            with torch.no_grad():
                # x_0 = torch.zeros_like(x)
                # x_1 = torch.ones_like(x)
                # idx_x = torch.topk(x, np.absolute(self.k), dim=1, largest = np.sign(self.k))[1]
                # mask_x = x_0.scatter_(1, idx_x, x_1)

                proto_0 = torch.zeros_like(targets)
                proto_1 = torch.ones_like(targets)
                idx_targets = torch.topk(targets, self.magk, dim=1, largest = self.largest)[1]
                mask_targets = proto_0.scatter_(1, idx_targets, proto_1)

                mask_x = mask_targets[labels]

            targets_ = targets*mask_targets
            x_ = x*mask_x
        else:
            targets_ = targets
            x_ = x

        if self.protonorm:
            targets_ = F.normalize(targets_)
            x_ = F.normalize(x_)


        #distmat is summation of all 1024 elements squared, expanded to [batch, 10]
        distmat = torch.pow(x_, 2).sum(dim=1, keepdim=True).expand(batch_size, numT) + \
                  torch.pow(targets_, 2).sum(dim=1, keepdim=True).expand(numT, batch_size).t()
        distmat.addmm_(x_, targets_.t(), beta=1, alpha=-2)
        distmat.clamp_(1e-6,1e6)
        #distmat = torch.sqrt(distmat)     # [batch_size, numT]

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))   #[batch_size, num_classes]

        # if self.cent_per_class > 1:
        #     y = torch.chunk(distmat, self.cent_per_class, dim=1)    #should give a [batch_size, num_classes] chunk
        #     res = y[0]*mask
        
        #     for i in range(1,self.cent_per_class):
        #         res = torch.minimum(res,y[i]*mask)
        # else:
       
        res = mask*distmat

        res = torch.max(res, dim=1)[0]

        #if self.k==0:
        #    res /= 512
        #else:
        #    res /= self.k

        loss = res.mean()

        #print (loss)

        if self.hsphere:
            mags = torch.linalg.norm(x_,dim=1)
            with torch.no_grad():
                mu = mags.mean()
            hloss = 0.5*((mags-mu.expand_as(mags))**2).mean()
            loss += hloss

        if self.psi <=0.0:
            return loss
        
        if self.protonorm:
            targets_norm = targets_
        else:
            targets_norm = F.normalize(targets_)

        #till 0129 am    
        #prox_sim  = targets_norm @ targets_norm.t()
        #prox_sim = (torch.triu(prox_sim, diagonal=1))**2

        #loss_sim = prox_sim.mean()

        #loss_total = (1/1.414)*loss + self.psi*loss_sim

        #startin 0129 pm
        prox_sim = targets_norm @ targets_norm.t()
        prox_sim.fill_diagonal_(0)
        prox_sim = prox_sim**2
        #maxes = torch.max(prox_sim, dim=1)[0]

        if self.maxmean:
            maxes = torch.topk(prox_sim,k=self.kprox,dim=1,sorted=False)[0]
            #print (maxes)
            if self.topkprox > 0:
                loss_sim = (torch.topk(maxes,k=self.topkprox,dim=0,sorted=False)[0]).mean()
            else:
                loss_sim = maxes.mean()
            #prox_sim_no_diag = (torch.triu(prox_sim, diagonal=1) + torch.tril(prox_sim, diagonal=-1))**2
        else:
            means = torch.mean(prox_sim,dim=1)
            loss_sim = (torch.topk(means,k=self.kprox,dim=0,sorted=False)[0]).mean()

        loss += self.psi*loss_sim


        #if self.decay_pow > 0.0:
        #    loss -= self.decay_const*(torch.linalg.norm(targets_, 2, dim=1)**self.decay_pow).mean()

        return loss


def prox_loss(model,
                image_model,
                device,
                x,
                y,
                optimizer,
                par_images_opt,
                parServant=0,
                beta=0.03,
                cpc = 1,
                k=0,
                decay_pow=0.0,
               decay_const=1.0,
                transformDict={},
                **kwargs):

    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    #criterion_prox = nn.KLDivLoss(reduction='batchmean')
    #self, device, num_classes=10, cent_per_class=1, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):

    criterion_prox = Proximity(device=device,
                               num_classes=kwargs['num_classes'],
                               cent_per_class=cpc,
                               k=k,
                               protonorm=kwargs['proto_norm'],
                               psi=kwargs['psi'],
                               kprox=kwargs['kprox'],
                               maxmean=kwargs['maxmean'],
                               proxpwr=kwargs['proxpwr'],
                               topkprox=kwargs['topkprox'],
                               hsphere=kwargs['hsphere'])

    
    criterion_kl = kldivproto(device=device, num_classes=kwargs['num_classes'])
    #criterion_cos = ProxCos(device=device)
    #model.eval()
    #batch_size = len(x_natural)
    # generate adversarial example

    model.train()

    # zero gradient
    optimizer.zero_grad()

    if beta > 0:
        if not kwargs['latent_proto']:
            par_images_opt_transform = transformDict['proto_no_norm'](par_images_opt.clone().detach())
            par_images_opt.data = par_images_opt_transform.data
            #par_images_opt_norm = (par_images_opt - protoMean) / protoStd
            par_images_opt = transformDict['norm'](par_images_opt)

    # with torch.no_grad():
        
    #     if kwargs['augmix']:

    #         x_int = x_natural.clone().detach()*255
    #         x_int = x_int.to(torch.uint8)
    #         #author implementation of augmix includes normalization in it, not sure if Pytorch includes this
    #         # it appears from pytorch implementation... it does not include any normalization
    #         #print ("min augmix value in image is ", torch.min(transformDict['aug'](x_int.clone())))

    #         #x_aug1 = (transformDict['aug'](x_int.clone())/255.0).to(torch.float32).clamp(0.0,1.0)
    #         #x_aug2 = (transformDict['aug'](x_int.clone())/255.0).to(torch.float32).clamp(0.0,1.0)

    #         x_aug1 = (transformDict['aug'](x_int.clone()).to(torch.float32)/255.0).clamp(0.0,1.0)
    #         x_aug2 = (transformDict['aug'](x_int.clone()).to(torch.float32)/255.0).clamp(0.0,1.0)
    #         x_aug1 = transformDict['norm'](x_aug1)
    #         x_aug2 = transformDict['norm'](x_aug2)

    #     x_natural = transformDict['basic'](x_natural)
    #print (torch.mean(x))
    #print (torch.min(x))
    #print (torch.max(x))

    y = y.to(device)
    

    model.apply(set_bn_train)

    if kwargs['js_loss']:
        x_natural, x_aug1, x_aug2 = x[0].to(device), x[1].to(device), x[2].to(device)
        L2_aug1, logits_aug1 = model(x_aug1)
        L2_aug2, logits_aug2 = model(x_aug2)
    else:
        x_natural = x.to(device)

    L2_inp, logits = model(x_natural)


    #batchnorm stats tracked on forward pass only, dont do it for the par images
    model.apply(set_bn_eval)

    if beta > 0:
        if not kwargs['latent_proto']:
            if (parServant):
                #image_model.load_state_dict(model.state_dict())
                #image_model.eval()
                L2_img, logits_img = image_model(par_images_opt)
            else:
                L2_img, logits_img = model(par_images_opt)
        else:
            #normalization takes place in loss function
            L2_img = par_images_opt

            
    loss = F.cross_entropy(logits, y)

    #print (loss)
    
    #move same class data towards par_image centers
    if beta > 0:
        #print ("calc prox loss")
        #include augmix?

        if (kwargs['js_loss']):
            L2_cat = torch.cat((L2_inp,L2_aug1,L2_aug2),dim=0)
            #print (L2_cat.shape)
            #print (y.shape)
            y_cat = y.repeat(3)
            L2_static = L2_cat.clone().detach()
            if parServant:
                loss += beta*criterion_prox(L2_static, y_cat, L2_img)
            else:
                loss += beta*criterion_prox(L2_cat, y_cat, L2_img)
        else:
            L2_static = L2_inp.clone().detach()
            if parServant:
                loss += beta*criterion_prox(L2_static, y, L2_img)
            else:
                loss += beta*criterion_prox(L2_inp, y, L2_img)

    # if klmatch:
    #     loss += gamma*criterion_kl(F.softmax(logits_img,dim=1),y,F.log_softmax(logits,dim=1))

    if kwargs['js_loss']:
        #JS-Divergence
        
        p_clean, p_aug1, p_aug2 = F.softmax(
          logits, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)
        
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3 
        
    return loss


def w_prox_loss(model,
                device,
                x,
                y,
                weights,
                optimizer,
                par_images_opt,
                beta=0.03,
                k=0,
                transformDict={},
                **kwargs):

    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    #criterion_prox = nn.KLDivLoss(reduction='batchmean')
    #self, device, num_classes=10, cent_per_class=1, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):
    #def __init__(self, device, num_classes=10, k=0, protonorm=0, psi=0.0, kprox=1, maxmean=1, proxpwr=1.0, topkprox=0, hsphere=0):

    criterion_prox = WeightedProximity(device=device,
                               num_classes=kwargs['num_classes'],
                               k=k,
                               protonorm=kwargs['proto_norm'],
                               psi=kwargs['psi'],
                               kprox=kwargs['kprox'],
                               maxmean=kwargs['maxmean'],
                               proxpwr=kwargs['proxpwr'],
                               topkprox=kwargs['topkprox'],
                               hsphere=kwargs['hsphere'])

    
    model.train()

    # zero gradient
    optimizer.zero_grad()

    if beta > 0:
        if not kwargs['latent_proto']:
            par_images_opt_transform = transformDict['proto_no_norm'](par_images_opt.clone().detach())
            par_images_opt.data = par_images_opt_transform.data
            #par_images_opt_norm = (par_images_opt - protoMean) / protoStd
            par_images_opt = transformDict['norm'](par_images_opt)

    y = y.to(device)
    weights= weights.to(device)

    model.apply(set_bn_train)

    #print (y)
    #print (torch.mean(x))
    #print (torch.min(x))
    #print (torch.max(x))

    x_natural = x.to(device)
    L2_inp, logits = model(x_natural)

    #batchnorm stats tracked on forward pass only, dont do it for the par images
    model.apply(set_bn_eval)

    if beta > 0:
        #print ("A")
        if not kwargs['latent_proto']:
            L2_img, logits_img = model(par_images_opt)
        else:
            #normalization takes place in loss function
            L2_img = par_images_opt

    if kwargs['wxent']:
        #print ("B")
        loss = (weights*F.cross_entropy(logits,y, reduction='none')).mean()
    else:
        #print ("C")
        loss = F.cross_entropy(logits, y)

    #print (loss)
    
    #move same class data towards par_image centers
    if beta > 0:
        #print ("D")
        loss += beta*criterion_prox(L2_inp, y, L2_img, weights)

    #print (loss)

        
    return loss
