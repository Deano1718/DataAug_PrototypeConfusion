
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from random import randint, choice, uniform, sample

from torchvision import datasets, transforms
from utils import *

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack















    # with torch.no_grad():

    #     x, y, w, x_id = get_split_points_weighted(dloader,**kwargs)

    #     for lbl in range(targets.shape[0]):

    #         new_images = []
    #         new_image_labels = []
    #         class_set = CustomDataSet(xl[lbl % num_classes],yl[lbl % num_classes])
    #         img_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=False)

    #         model.multi_out = 0

    #         alpha_first = []
    #         alpha_cur = []
    #         alpha_range = []

    #         for batch_idx, data in enumerate(img_loader):
            
    #             X, Y = data[0].cpu(), data[1].cpu()
    #             bs = X.size(0)



def expand_data(args, model, device, dloader, par_tens, cur_epoch, transformDict={}, confusionList=[], **kwargs):

    model.eval()
    par_tens = par_tens.clone().cpu()
    num_classes = kwargs['num_classes']

    
    # if kwargs["proto_aug"]:
    #     protolist = []
    #     if "crop" in kwargs["proto_aug"]:
    #         protolist.append(transforms.RandomCrop(32, padding=4))
    #     if "flip" in kwargs["proto_aug"]:
    #         protolist.append(transforms.RandomHorizontalFlip(p=0.5))
    #     if "invert" in kwargs["proto_aug"]:
    #         protolist.append(transforms.RandomInvert(p=0.5))                                                                                                                                                          
    #     transformProto = transforms.Compose(protolist)
    
    with torch.no_grad():

        xl, yl = get_split_points(dloader,**kwargs)
        #new_images = []
        #new_image_labels = []
        new_image_tensors = []
        new_label_tensors = []

        if (args.inter_mix and (((cur_epoch-args.expand_data_epoch) % args.expand_interval) == 0)):
            
            for lbl in range(par_tens.shape[0]):
                #print (len(xl[lbl]))
                #print (xl[lbl].shape)
                #print (yl[lbl].shape)
                new_images = []
                new_image_labels = []
                class_set = CustomDataSet(xl[lbl % num_classes],yl[lbl % num_classes])
                img_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=False)

                model.multi_out = 0

                alpha_first = []
                alpha_cur = []
                alpha_range = []

                for batch_idx, data in enumerate(img_loader):
                
                    X, Y = data[0].cpu(), data[1].cpu()
                    bs = X.size(0)

                    alpha_first_batch = torch.ones_like(Y, dtype=torch.float)
                    alpha_center_batch = torch.zeros_like(Y, dtype=torch.float)
                    alpha_cur_batch = torch.zeros_like(Y, dtype=torch.float)

                    if kwargs['proto_aug']:
                        par_tens_cur = transformDict['proto_no_norm'](par_tens[lbl].clone().unsqueeze(0))
                    else:
                        par_tens_cur = par_tens[lbl].clone().unsqueeze(0)

                    for alpha in range(1,20):

                        alpha /= 20.0

                        Xalpha = (1.0-alpha)*par_tens_cur.expand(bs,-1,-1,-1) + alpha*X.clone()
                        Xalphafwd = Xalpha.clone().to(device)

                        Xalphafwd = transformDict['norm'](Xalphafwd)
                        
                        Z = model(Xalphafwd)

                        Z=Z.cpu()

                        #make prediction, do not need softmax
                        Yp = Z.data.max(dim=1)[1]  # get the index of the max for each batch sample

                        if (torch.any(Yp != Y)):
                            alpha_cur_batch[Yp != Y] = alpha
                            alpha_first_batch[Yp != Y] = torch.minimum(alpha_first_batch[Yp != Y], alpha_cur_batch[Yp != Y])
                            alpha_center_batch[Yp != Y] = (alpha_first_batch[Yp != Y] + alpha_cur_batch[Yp != Y]) / 2.0

            
                    inds = torch.nonzero(alpha_center_batch, as_tuple=True)[0]
                    #print (inds.shape)
                    #print (X[inds].shape)
                    #print ((1.0-alpha_center_batch[inds]).shape)
                    #print (par_tens[lbl].clone().unsqueeze(0).expand(len(inds),-1,-1,-1).shape)
                    if (len(inds)>0):
                        alpha_tens = alpha_center_batch[inds].unsqueeze(1).unsqueeze(1).unsqueeze(1).clone()
                        #print (alpha_tens.shape)
                        #a = alpha_tens*X[inds]
                        #b = (1.0-alpha_tens)
                        #c = (par_tens_cur.expand(len(inds),-1,-1,-1))
                        #print (a.shape)
                        #print (b.shape)
                        #print (c.shape)
                        Xalpha_new = alpha_tens*X[inds].clone() +(1.0-alpha_tens)*(par_tens_cur.expand(len(inds),-1,-1,-1))
                        Yalpha_new = Y[inds].clone()
                        new_images.append(Xalpha_new)
                        new_image_labels.append(Yalpha_new)

                if len(new_images)>0:
                    new_image_tensors.append(torch.cat(new_images,dim=0).clone().cpu())
                    new_label_tensors.append(torch.cat(new_image_labels,dim=0).clone().cpu())


    

    if args.mixup and (((cur_epoch-args.expand_data_epoch)%args.mix_interval)==0):
        print ("conducting mixup expansion")
        new_images = []
        new_image_labels = []
        model.multi_out = 0
        #DeepFool test on parametric images
        
        par_image_labels = torch.remainder(torch.arange(par_tens.shape[0]),num_classes).long().to(device)
        #print ("par_image_labels ", par_image_labels)
        preprocessing = dict(mean=transformDict['mean'], std=transformDict['std'], axis=-3)
        fmodel=PyTorchModel(model, bounds=(0,1),preprocessing=preprocessing)            
        attack = L2DeepFoolAttack(steps=100)
        
        par_tens_cuda = par_tens.clone().detach().to(device)
        raw, X_new_torch, is_adv = attack(fmodel, par_tens_cuda, par_image_labels, epsilons = 25.0 )


        print ("DF classes found")
        #print ("X new torch min", torch.min(X_new_torch))
        pairs = []
        
        with torch.no_grad():

            X_new_torch = transformDict['norm'](X_new_torch)

            #print ("X new torch min after norm", torch.min(X_new_torch))

            Z = model(X_new_torch)
            Yp_adv =  Z.data.max(dim=1)[1]
            #print (Yp_adv)
            for i in range(len(Yp_adv)):
                if ((i % num_classes) != Yp_adv[i].item()):
                    # (prototype, opposing class)
                    if ((i,Yp_adv[i].item()) not in confusionList):
                        pairs.append((i,Yp_adv[i].item()))
                    if ((Yp_adv[i].item(),i) not in confusionList):
                        pairs.append((Yp_adv[i].item(),i))   #the reflexive case

            confusionList.extend(pairs)
                
        for tup in confusionList:
            #get orig training set clean examples from "opposing class"
            # xl is a list with each entry being the tensor of examples from class_<index>
            # yl is a list with each entry being the tensor of labels from class_<index>
            class_set = CustomDataSet(xl[tup[1]],yl[tup[1]])

            img_loader = torch.utils.data.DataLoader(class_set, batch_size=200, shuffle=False)

            for batch_idx, data in enumerate(img_loader):

                X, Y = data[0].cpu(), data[1].cpu()
                bs = X.size(0)

                if kwargs['proto_aug']:
                    par_tens_cur = transformDict['proto_no_norm'](par_tens[tup[0]].clone().unsqueeze(0))
                else:
                    par_tens_cur = par_tens[tup[0]].clone().unsqueeze(0)

                Xalpha = (1.0-args.alpha_mix)*par_tens_cur.expand(bs,-1,-1,-1) + args.alpha_mix*X.clone()


                Xalphafwd = Xalpha.clone().to(device)

                Xalphafwd = transformDict['norm'](Xalphafwd)
                
                Z = model(Xalphafwd)

                Z=Z.cpu()

                #make prediction, do not need softmax
                Yp = Z.data.max(dim=1)[1]  # get the index of the max for each batch sample

                if (torch.any(Yp != Y)):
                    new_images.append(Xalpha[Yp!=Y].clone())
                    new_image_labels.append(Y[Yp!=Y].clone())

        if len(new_images)>0:
            new_image_tensors.append(torch.cat(new_images,dim=0).clone().cpu())
            new_label_tensors.append(torch.cat(new_image_labels,dim=0).clone().cpu())


            

    #after all labels
    Xalpha_new = torch.cat(new_image_tensors,dim=0)
    Yalpha_new = torch.cat(new_label_tensors, dim=0)
    #print ("Xalpha_new shape ", Xalpha_new.shape)
    #print ("Yalpha_new shape ", Yalpha_new.shape)
    new_dataset = CustomDataSet(Xalpha_new,Yalpha_new)

    return new_dataset, confusionList



def recompute_confusion(model, device, prototype_images, cur_confusion, num_classes, transformDict):
    model.multi_out = 0
    model.eval()
    #DeepFool test on parametric images
    
    par_image_labels = torch.remainder(torch.arange(prototype_images.shape[0]),num_classes).long().to(device)
    #print ("par_image_labels ", par_image_labels)
    preprocessing = dict(mean=transformDict['mean'], std=transformDict['std'], axis=-3)
    fmodel=PyTorchModel(model, bounds=(0,1),preprocessing=preprocessing)            
    attack = L2DeepFoolAttack(steps=100)
    
    par_tens_cuda = prototype_images.clone().detach().to(device)
    raw, X_new_torch, is_adv = attack(fmodel, par_tens_cuda, par_image_labels, epsilons = 25.0 )


    print ("DF classes found")
    #print ("X new torch min", torch.min(X_new_torch))
    
    
    with torch.no_grad():

        X_new_torch = transformDict['norm'](X_new_torch)

        Z = model(X_new_torch)
        Yp_adv =  Z.data.max(dim=1)[1]
        #print (Yp_adv)
        for i in range(len(Yp_adv)):
            if ((i % num_classes) != Yp_adv[i]):
                # (prototype, opposing class)
                if (Yp_adv[i] not in cur_confusion[i]):
                    cur_confusion[i%num_classes].append(Yp_adv[i].cpu().item())

                if ( (i%num_classes) not in cur_confusion[Yp_adv[i]]):
                    cur_confusion[Yp_adv[i].cpu().item()].append(i%num_classes)   #the reflexive case

    return cur_confusion




class ConfusionAugWrapper(torch.nn.Module):
    def __init__(self, preprocess, proto_preprocess, num_classes, prototypes, confusionHash, js_loss=False, m_ave = 0.8,
                 final_process=True, pipelength=1, confusionMode=0, mode0rand=0, window=[], counts=[], imsize=32):
        #self.dataset = dataset
        self.preprocess = preprocess
        print (self.preprocess)
        #prototype preprocess should at least have flipping
        if proto_preprocess:
            self.proto_preprocess = transforms.Compose(proto_preprocess)
        else:
            self.proto_preprocess = []
        #in the (final_process && js_loss case, ConfusionAug will receive a 3-tuple from previous dataset (like augmix))
        self.js_loss = js_loss
        #confusionList is list of lists
        self.confusionHash = confusionHash
        #keep static batch of prototype tensors
        self.prototypes = prototypes
        self.m_ave = m_ave
        self.num_classes = num_classes
        #final_process is False if use in conjunction with AugMix or Prime in future
        self.final_process = final_process
        self.active = 0
        self.pipelength= pipelength
        self.confusionMode = confusionMode   #0 (mode0,mode0), 1 (mode1,mode1), 2 (mode0, mode1), 3 (random,random)
        self.mode0rand=mode0rand
        self.window_size = window #[3, 5]
        self.counts = counts #[19, 7]
        self.inds = len(counts)
        self.imsize = imsize

    def update_confusion(self, conf):
        self.confusionHash = conf

    def update_proto(self, tens):
        self.prototypes = tens.clone().detach()

    def augment(self, tens, y, mode):
        #this method is only used when confusion is final process after prime or augmix
        #its preprocess will always include normalization
        #receives a tensor
        aug_tens=[]

        if self.active:
            if self.confusionMode==1:
                mode = 1
            if self.confusionMode==0:
                mode = 0
                
            for t in range(tens.shape[0]):
                if self.confusionMode==3:
                    mode=np.random.randint(0,2)
                if self.confusionHash[y[t]]:
                    confusor1 = np.random.choice(self.confusionHash[y[t]])
                else:
                    confusor1 = np.random.randint(0,num_classes-1)
                aug_tens.append(self.preprocess(self.confusion_mix(tens[t], confusor1, mode=mode)))
            aug_tens = torch.stack(aug_tens,dim=0)
            return aug_tens
        
        else:
            #confusion is final process in chain, should normalize data even if not active
            return self.preprocess(tens)

    def preprocess_only(self, tens):
        return self.preprocess(tens)
                

    def forward(self,x,y):

        if self.active:
            mode = np.random.randint(0,2)
            if self.js_loss:
                if self.confusionHash[y]:
                    confusor1 = np.random.choice(self.confusionHash[y])
                    confusor2 = np.random.choice(self.confusionHash[y])
                else:
                    confusor1 = np.random.randint(0,num_classes-1)
                    confusor2 = np.random.randint(0,num_classes-1)
            else:
                if self.confusionHash[y]:
                    confusor1 = np.random.choice(self.confusionHash[y])
                else:
                    confusor1 = np.random.randint(0,num_classes-1)

            if self.final_process:
                if self.js_loss and self.pipelength>1:
                    im_tuple = (self.preprocess(x[0]), 
                        self.preprocess(self.confusion_mix(x[1],  confusor1, mode=0)),
                        self.preprocess(self.confusion_mix(x[2], confusor2, mode=1)))
                    return im_tuple, y
                elif self.js_loss:
                    im_tuple = (self.preprocess(x),
                        self.preprocess(self.confusion_mix(x,  confusor1, mode=0)),
                        self.preprocess(self.confusion_mix(x,  confusor2, mode=1)))
                    return im_tuple, y
                else:
                    return self.preprocess(self.confusion_mix(x, confusor1, mode)), y
            else:
                if self.js_loss:
                    im_tuple = (x, self.confusion_mix(x, confusor1, mode=0), self.confusion_mix(x, confusor2, mode=1))
                    return im_tuple, y
                else:
                    return confusion_mix(x, confusor1, mode), y
        else:
            if self.final_process:
                if self.js_loss and self.pipelength>1:
                    im_tuple = (self.preprocess(x[0]), 
                        self.preprocess(x[1]),
                        self.preprocess(x[2]))
                    return im_tuple, y
                elif self.js_loss:
                    im_tuple = (self.preprocess(x),
                        self.preprocess(x),
                        self.preprocess(x))
                    return im_tuple, y
                else:
                    return self.preprocess(x), y
            else:
                if self.js_loss:
                    im_tuple = (x, x, x)
                    return im_tuple, y
                else:
                    return x, y

    def confusion_mix(self,image, confusor, mode):
        """Perform confusionmix augmentations and compute mixture.
        Args:
          image: input image tensor
          proto_preprocess: Preprocessing function which affects prototypes during mixing.
        Returns:
          im: Augmented and mixed tensor image.
        """
        with torch.no_grad():

            im = image.clone()
            
            if self.proto_preprocess:
                confusing_prototype = self.proto_preprocess(self.prototypes[confusor].clone())
            else:
                confusing_prototype = self.prototypes[confusor].clone()

            if mode:
                m = np.random.uniform(self.m_ave-0.05, self.m_ave+0.05)
                return (1-m)*confusing_prototype + m*im

            else:
                wind = 5
                cnt = 7
                if self.mode0rand:
                    ind = np.random.randint(0,self.inds)
                    wind = self.window_size[ind]
                    wind2 = int(wind/2)
                    cnt = self.counts[ind]
                #randomlist = np.random.random_integers(low=1,high=31, size=10)
                randomrow = np.random.random_integers(low=wind2,high=(self.imsize-wind2), size=cnt)
                randomcol = np.random.random_integers(low=wind2,high=(self.imsize-wind2), size=cnt)
                for i,num in enumerate(randomrow):
                    #images.view(-1,3,-1)[:,:,num-4:num+5]
                    #images[:,:,num-1:num+2,num-1:num+2] = y[:,:,num-1:num+2,num-1:num+2]
                    im[:,randomrow[i]-wind2:randomrow[i]+wind2+1,randomcol[i]-wind2:randomcol[i]+wind2+1] = confusing_prototype[:,randomrow[i]-wind2:randomrow[i]+wind2+1,randomcol[i]-wind2:randomcol[i]+wind2+1]
                return im


class ConfusionAugDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, proto_preprocess, num_classes, prototypes, confusionHash, js_loss=False, m_ave = 0.8,
                 final_process=True, pipelength=1, confusionMode=2, mode0rand=0, window=[], counts=[], imsize=32):
        self.dataset = dataset
        self.preprocess = preprocess
        #prototype preprocess should at least have flipping
    
        if proto_preprocess:
            self.proto_preprocess = transforms.Compose(proto_preprocess)
        else:
            self.proto_preprocess = []

        #in the (final_process && js_loss case, ConfusionAug will receive a 3-tuple from previous dataset (like augmix))
        self.js_loss = js_loss
        #confusionList is list of lists
        self.confusionHash = confusionHash
        #keep static batch of prototype tensors
        self.prototypes = prototypes
        self.m_ave = m_ave
        self.num_classes = num_classes
        #final_process is False if use in conjunction with AugMix or Prime in future
        self.final_process = final_process
        self.pipelength=pipelength
        self.active = 0
        self.confusionMode = confusionMode   #0 (mode0,mode0), 1 (mode1,mode1), 2 (mode0, mode1) 3 (random,random)         
        self.mode0rand=mode0rand
        self.window_size = window #[3, 5]
        self.counts = counts #[19, 7] 
        self.inds = len(counts)
        self.imsize=32



    def update_confusion(self,conf):
        self.confusionHash = conf


    def update_proto(self, tens):
        self.prototypes = tens.clone().detach().cpu()


    def __getitem__(self, i):
        x, y = self.dataset[i]

        if self.active:
            modes = [0,1]
            if self.confusionMode == 3:
                modes[0] = np.random.randint(0,2)
                modes[1] = np.random.randint(0,2)
            if self.confusionMode == 1:
                modes[0] = 1
            if self.confusionMode == 0:
                modes[1] = 0
                
            if self.js_loss:
                if self.confusionHash[y]:
                    confusor1 = np.random.choice(self.confusionHash[y])
                    confusor2 = np.random.choice(self.confusionHash[y])
                else:
                    confusor1 = np.random.randint(0,num_classes-1)
                    confusor2 = np.random.randint(0,num_classes-1)
            else:
                if self.confusionHash[y]:
                    confusor1 = np.random.choice(self.confusionHash[y])
                else:
                    confusor1 = np.random.randint(0,num_classes-1)

            if self.final_process:
                if self.js_loss and self.pipelength>1:
                    im_tuple = (self.preprocess(x[0]), 
                        self.preprocess(self.confusion_mix(x[1],  confusor1, mode=modes[0])),
                        self.preprocess(self.confusion_mix(x[2],  confusor2, mode=modes[1])))
                    return im_tuple, y
                elif self.js_loss:
                    im_tuple = (self.preprocess(x),
                        self.preprocess(self.confusion_mix(x,  confusor1, mode=modes[0])),
                        self.preprocess(self.confusion_mix(x,  confusor2, mode=modes[1])))
                    return im_tuple, y
                else:
                    return self.preprocess(self.confusion_mix(x, confusor1, modes[0])), y
            else:
                if self.js_loss:
                    im_tuple = (x, self.confusion_mix(x, confusor1, mode=modes[0]), self.confusion_mix(x, confusor2, mode=modes[1]))
                    return im_tuple, y
                else:
                    return confusion_mix(x, confusor1, modes[0]), y
        else:
            if self.final_process:
                if self.js_loss and self.pipelength>1:
                    im_tuple = (self.preprocess(x[0]), 
                        self.preprocess(x[1]),
                        self.preprocess(x[2]))
                    return im_tuple, y
                elif self.js_loss:
                    im_tuple = (self.preprocess(x),
                        self.preprocess(x),
                        self.preprocess(x))
                    return im_tuple, y
                else:
                    return self.preprocess(x), y
            else:
                if self.js_loss:
                    im_tuple = (x,x,x)
                    return im_tuple, y
                else:
                    return x, y


    def __len__(self):
        return len(self.dataset)




    def confusion_mix(self,image,confusor, mode):
        """Perform confusionmix augmentations and compute mixture.
        Args:
          image: input image tensor
          proto_preprocess: Preprocessing function which affects prototypes during mixing.
        Returns:
          im: Augmented and mixed tensor image.
        """
        with torch.no_grad():

            im = image.clone()
            
            if self.proto_preprocess:
                confusing_prototype = self.proto_preprocess(self.prototypes[confusor].clone())
            else:
                confusing_prototype = self.prototypes[confusor].clone()


            if mode:
                m = np.random.uniform(self.m_ave-0.05, self.m_ave+0.05)
                return (1-m)*confusing_prototype + m*im

            else:
                wind = 5
                cnt = 7
                if self.mode0rand:
                    ind = np.random.randint(0,self.inds)
                    wind = self.window_size[ind]
                    wind2 = int(wind/2)
                    cnt = self.counts[ind]
                #randomlist = np.random.random_integers(low=1,high=31, size=10)
                randomrow = np.random.random_integers(low=wind2,high=(self.imsize-wind2), size=cnt)
                randomcol = np.random.random_integers(low=wind2,high=(self.imsize-wind2), size=cnt)
                for i,num in enumerate(randomrow):
                    #images.view(-1,3,-1)[:,:,num-4:num+5]
                    #images[:,:,num-1:num+2,num-1:num+2] = y[:,:,num-1:num+2,num-1:num+2] 
                    im[:,randomrow[i]-wind2:randomrow[i]+wind2+1,randomcol[i]-wind2:randomcol[i]+wind2+1] = confusing_prototype[:,randomrow[i]-wind2:randomrow[i]+wind2+1,randomcol[i]-wind2:randomcol[i]+wind2+1]
                return im
