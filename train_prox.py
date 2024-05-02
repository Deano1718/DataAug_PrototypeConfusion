from __future__ import print_function
import os
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

from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
#from torch.utils.tensorboard import SummaryWriter

#from models.wideresnet import *
from models.resnet import *
from models.simple import *
from models.densenet import *
from models.resnext import *
from models.allconv import *
from models.wideresnet import *
from loss_utils import *
from utils import *

# import augmentations
# from color_jitter import *
# from diffeomorphism import *
# from rand_filter import *

# from torch.distributions import Dirichlet, Beta
# from einops import rearrange, repeat
# from opt_einsum import contract

from utils_confusion import *
from utils_augmix import *
from utils_prime import *
#from trades import trades_loss
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack
#from create_data import compute_smooth_data, merge_data, CustomDataSet

#from robustness.datasets import CustomImageNet
#from robustness.datasets import DATASETS, DataSet, CustomImageNet
#import smoothers


parser = argparse.ArgumentParser(description='PyTorch CIFAR + proximity training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--beta', default=0.03, type=float,
                    help='loss weight for proximity')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='../ProtoRuns/model-cifar10-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model',default="ResNet18",
                    help='network to use')
parser.add_argument('--restart',default=0, type=int,
                    help='restart training, make sure to specify directory')
parser.add_argument('--restart-epoch',default=0, type=int,
                    help='epoch to restart from')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--par-servant', default=0, type=int,
                    help='whether normalization is learnable')
parser.add_argument('--par-sparse', default=0, type=int,
                    help='force L1 sparsity on prototype images')
parser.add_argument('--par-zeta', default=0.01, type=float,
                    help='L1 sparsity multiplier on prototype images')
parser.add_argument('--expand-data-epoch', default=0, type=int,
                    help='start mixing data with misclassified combinations with parametric images')
parser.add_argument('--expand-interval', default=5, type=int,
                    help='number of epochs to wait before re-expanding')
parser.add_argument('--kldiv', default=0, type=int,
                    help='enforce kldiv match between prototype and examples')
parser.add_argument('--gamma', default=0.0, type=float,
                    help='mult loss for kldiv match')
parser.add_argument('--mixup', default=0, type=int,
                    help='augment data with mixup par class x to examples not x')
parser.add_argument('--alpha-mix', default=0.7, type=float,
                    help='alpha for mixup')
parser.add_argument('--mix-interval',default=3, type=int,
                    help='how often to intra class mix')
parser.add_argument('--par-grad-mult', default=10.0, type=float,
                    help='boost image gradients if desired')
parser.add_argument('--par-grad-clip', default=0.01, type=float,
                    help='max magnitude per update for proto image updates')
parser.add_argument('--class-centers', default=1, type=int,
                    help='number of parametric centers per class')
parser.add_argument('--dataset', default="CIFAR10",
                    help='which dataset to use, CIFAR10, CIFAR100, IN100')
parser.add_argument('--image-train', default=0, type=int,
                    help='train parametric images on frozen model')
parser.add_argument('--norm-data', default=0, type=int,
                    help='normalize data')
parser.add_argument('--anneal', default="stairstep", 
                    help='type of LR schedule stairstep, cosine, or cyclic')
parser.add_argument('--inter-mix', default=0, type=int,
                    help='fill in holes within same class')
#parser.add_argument('--augmix', default=0, type=int,
#                     help='use augmix data augmentation')
#parser.add_argument('--prime', default=0, type=int,
#                     help='use PRIME data augmentation')
#parser.add_argument('--confusionmix', default=0, type=int,
#                     help='use confusionmix data augmentation')
parser.add_argument('--js-loss', default=1, type=int,
                    help='use jensen shannon divergence for augmix')
parser.add_argument('--pipeline', nargs='+',default=[],
                    help='augmentation pipeline')
parser.add_argument('--grad-clip', default = 1, type=int,
                    help='clip model weight gradients by 0.5')
parser.add_argument('--confusion-mode', default = 2, type=int,
                    help='0 = (mode0,mode0), 1 = (mode1,mode1), 2= (mode0,mode1) 3= (random,random)')
parser.add_argument('--mode0rand', default = 0, type=int,
                    help='randomly switch between window crop size 3 and 5 in mode 0')
parser.add_argument('--channel-norm', default = 1, type=int,
                    help='normalize each channel by training set mean and std')
parser.add_argument('--channel-swap', default = 0, type=int,
                    help='randomly permute channels augmentation')
parser.add_argument('--window', nargs='+', default=[], type=int,
                    help='possible windows for cutouts')
parser.add_argument('--counts', nargs='+', default=[], type=int,
                    help='possible counts for windows')
parser.add_argument('--proto-layer', default = 5, type=int,
                    help='after which block to compute prototype loss')
parser.add_argument('--proto-pool', default ="none",
                    help='whether to adaptive pool proto vector to Cx1 and how')
parser.add_argument('--proto-norm', default = 0, type=int,
                    help='normalize vectors before prototype loss computed')
parser.add_argument('--proto-aug', nargs='+',default=[],
                    help='augmentations for prototype image')
parser.add_argument('--k', default=0, type=int,
                    help='consider only top +k or bottom -k elements of prototype vector, sorting based on prototype')
#parser.add_argument('--decay_pow', default=0.0, type=float,
#                    help='reduce loss by magnitude of prototype')
#parser.add_argument('--decay_const', default=1.0, type=float,
#                    help='reduce loss by magnitude of prototype')
#parser.add_argument('--renorm-prox', default=0, type=int,
#                    help='set to 1 if proto-norm =0')
parser.add_argument('--psi', default=1.0, type=float,
                    help='weight for proxcos contribution, multiplied by beta')
parser.add_argument('--latent-proto', default=0, type=int,
                    help='whether prototypes should be held in latent space as opposed to image space')
parser.add_argument('--kprox', default=1, type=int,
                    help='topk of each row to consider in proto cosine sim loss')
parser.add_argument('--maxmean', default=1, type=int,
                    help='if 1, will use topk maxes from each row, if 0, topk means from cossim matrix')
parser.add_argument('--proxpwr', default=1.0, type=float,
                    help='power of the L2 dist on data to prototype')
parser.add_argument('--topkprox', default=0, type=int,
                    help='if not 0, will select only topk maxes from kprox selection ie top10 of top5 maxes')
parser.add_argument('--hsphere', default=0, type=int,
                    help='shrink variance on magnitudes to speed convergence')
parser.add_argument('--droprate', nargs='+', default=[0.0],
                    help='include dropout')

parser.add_argument('--model-scale', default=1.0, type=float,
                    help='model scale')

# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--all-ops',
    default=1,
    type=int,
    help='Turn on all operations (+brightness,contrast,color,sharpness).')



args = parser.parse_args()

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
#kwargsUser['augmix'] = args.augmix
#kwargsUser['prime'] = args.prime
kwargsUser['js_loss'] = args.js_loss
kwargsUser['proto_aug'] = args.proto_aug
kwargsUser['pipeline'] = args.pipeline
kwargsUser['augmix'] = "augmix" in args.pipeline
kwargsUser['prime'] = "prime" in args.pipeline
kwargsUser['confusion'] = "confusion" in args.pipeline
kwargsUser['pipelength'] = len(args.pipeline)
kwargsUser['proto_layer'] = args.proto_layer
kwargsUser['proto_pool'] = args.proto_pool
kwargsUser['proto_norm'] = args.proto_norm
#kwargsUser['renorm_prox'] = args.renorm_prox
kwargsUser['psi']= args.psi
kwargsUser['latent_proto'] = args.latent_proto
kwargsUser['kprox'] = args.kprox
kwargsUser['maxmean'] = args.maxmean
kwargsUser['proxpwr'] = args.proxpwr
kwargsUser['topkprox'] = args.topkprox
kwargsUser['hsphere'] = args.hsphere
kwargsUser['droprate'] = args.droprate

assert (args.proto_pool in ['none','max','ave'])



# settings
if (args.model == "ResNet18"):
    network_string = "ResNet18"
elif (args.model =="LogNetBaseline"):
    network_string = "LogNet"
elif (args.model == "PreActResNet18"):
    network_string = "PreActResNet18"
elif ("WRN" in args.model):
    network_string = args.model
elif (args.model == "DenseNet"):
    network_string = "DenseNet"
elif (args.model == "ResNeXt"):
    network_string = "ResNeXt"
elif (args.model == "AllConv"):
    network_string = "AllConv"
else:
    print ("Invalid model architecture")
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string


if (not args.restart and not args.image_train):
    model_dir = ("{}_{}_beta_{}_scale_{}_pool_{}_norm_{}_{}".format("../ProtoRuns/model-{}".format(args.dataset),network_string,args.beta,args.model_scale,args.proto_pool,
        args.proto_norm,get_datetime()))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
else:
    model_dir = args.model_dir

if not args.image_train:    
    with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)


def train_image(args, model, device, train_loader, epoch, par_images, transformDict={},**kwargs):

    print ('Training images')
    criterion_prox = Proximity(device=device, num_classes=kwargs['num_classes'],cent_per_class=args.class_centers)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)


        _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)

        #print (transformDict['norm'])
        data = transformDict['norm'](data)
        _par_images_opt_norm = transformDict['norm'](_par_images_opt)
        #_par_images_opt_norm = transformDict['norm'](_par_images_opt.clone().detach())
        #_par_images_opt.data = _par_images_opt_norm.data

        L2_inp, logits = model(data)
        L2_img, logits_img = model(_par_images_opt_norm)

        loss = args.beta*criterion_prox(L2_inp, target, L2_img)

        loss.backward()


        with torch.no_grad():
            image_gradients = args.par_grad_mult*_par_images_opt.grad
            image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
            #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
            #print (torch.mean(image_gradients))
            #print ("image gradients are ", image_gradients)
            par_images.add_(-image_gradients)
            par_images.clamp_(0.0,1.0)
            _par_images_opt.grad.zero_()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))



def train(args, model, image_model, device, cur_loader, optimizer, epoch, par_images, scheduler=0.0, max_steps = 0, par_sparse=0, zeta=0.01, 
    PrimeWrapper=0.0, ConfuseWrapper=0.0, transformDict={}, flags=(), **kwargs):

    model.train()
    print ('Training model')

    glob_k = 0
    if (args.k != 0 and epoch > 0):
        glob_k = args.k


    for batch_idx, (data, target) in enumerate(cur_loader):

        #data, target = data.to(device), target.to(device)
        #print (batch_idx)
        #print (torch.min(data))
        #print ("mean ", MEAN)
        optimizer.zero_grad()

        if (not args.pipeline):
            data = transformDict['norm'](data)

        if kwargsUser['prime']:
            with torch.no_grad():
                #print (data.shape)
                #solo prime
                if flags[1] and kwargsUser['pipelength']==1:
                    data = data.to(device)
                    #apply Prime augmentations
                    data = PrimeWrapper(data)
                elif flags[1] and args.js_loss:
                    data = (data[0].to(device), data[1].to(device), data[2].to(device))
                    data = PrimeWrapper(data)
                else:
                    data = data.to(device)
                    data = PrimeWrapper(data)
                    #print (torch.max(data[0]))

                #data can be either single tensor or 3-tuple depending on js-loss
                #each tensors is [N,C,H,W]
                #print (data[0].shape)
                #print (data[1].shape)

                if flags[2]:
                    #confusion final process
                    if args.js_loss:
                        data[0] = ConfuseWrapper.preprocess_only(data[0])
                        data[1] = ConfuseWrapper.augment(data[1], target, mode=0)
                        data[2] = ConfuseWrapper.augment(data[2], target, mode=1)
                        #print (torch.max(data[0]))
                    else:
                        mode = np.random.randint(0,2)
                        data = ConfuseWrapper.augment(data,target,mode=mode)
                    


        # lets make par_servant useful again
        if (args.par_servant):
            image_model.load_state_dict(model.state_dict())
            image_model.eval()

        if args.beta > 0.0:
            _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)
        else:
            _par_images_opt = 0.0


        # calculate natural and proximity loss
        loss = prox_loss(model=model,
                            image_model=image_model,
                            device=device,
                            x=data,
                            y=target,
                            optimizer=optimizer,
                            par_images_opt = _par_images_opt,
                            parServant = args.par_servant,
                            beta=args.beta,
                         cpc=args.class_centers,
                         k = glob_k,
                         transformDict = transformDict,
                         **kwargs)

        if par_sparse:
            #l2 = torch.linalg.norm(_par_images_opt.view(10,-1),ord=2,dim=1)
            l1 = torch.linalg.norm(_par_images_opt.view(par_images.shape[0],-1),ord=1,dim=1)
            loss += zeta*torch.mean(l1)
            #loss += zeta*torch.mean(torch.sum(torch.abs(_par_images_opt).view(10,-1),dim=1),dim=0)

        loss.backward()


        if args.beta > 0.0:
            with torch.no_grad():
                if kwargsUser['latent_proto']:
                    latent_gradients = _par_images_opt.grad
                    par_images.add_(-latent_gradients)
                    par_images.clamp_(0.0,1e6)
                    _par_images_opt.grad.zero_()
                else:
                    image_gradients = args.par_grad_mult*_par_images_opt.grad
                    image_gradients.clamp_(-args.par_grad_clip,args.par_grad_clip)
                    #image_gradients =_par_images_opt.grad.clamp_(-0.002,0.002)
                    #print (torch.mean(image_gradients))
                    #print ("image gradients are ", image_gradients)
                    par_images.add_(-image_gradients)
                    par_images.clamp_(0.0,1.0)
                    _par_images_opt.grad.zero_()

        if (args.grad_clip):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

        optimizer.step()

        if args.anneal == "cyclic" or args.anneal == "cosine":
            if batch_idx < max_steps:
                scheduler.step()
                
        if kwargsUser['confusion']:

            if kwargsUser['augmix']:
                if (flags[2]):
                    cur_loader.dataset.update_proto(_par_images_opt)
                else:
                    cur_loader.dataset.dataset.update_proto(_par_images_opt)

            elif kwargsUser['prime']:
                if (flags[2]):
                    ConfuseWrapper.update_proto(_par_images_opt)
                else:
                    cur_loader.dataset.update_proto(_par_images_opt)
            else:
                cur_loader.dataset.update_proto(_par_images_opt)

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item()))


def eval_train(model, device, train_loader, transformDict):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, transformDict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= (0.5*args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75*args.epochs):
        lr = args.lr * 0.01
    # if epoch >= (0.9*args.epochs):
    #     lr = args.lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
#    # setup data loader
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if (args.dataset == "CIFAR10"):
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2471, 0.2435, 0.2616] 
    elif(args.dataset == "CIFAR100"):
        #MEAN = [0.5071, 0.4865, 0.4409]
        #STD = [0.2673, 0.2564, 0.2762]
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]
    elif  (args.dataset == "IN100"):
        #ImageNetFolder = "./Data_ImageNet/"
        #WordsFolder = "./Data_ImageNet/words/"
        MEAN = [0.5]*3
        STD = [0.5]*3
        if args.channel_norm:
            MEAN = [0.485, 0.456, 0.406]
            STD  = [0.229, 0.224, 0.225]

    else:
        print ("ERROR dataset not found")

    gen_transform_train = transforms.Compose([transforms.ToTensor()])
    #gen_transform_test = transforms.Compose([transforms.ToTensor()])

    #first augmentation in pipeline gets [Tensor, Flip, Crop] by default
    if args.dataset in ["CIFAR10","CIFAR100"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])
    elif args.dataset in ["IN100"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(256),
             transforms.CenterCrop(224)])
    else:
        print ("ERROR setting transforms")

    # if not args.augmix:
    #     train_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])
    # else:
    #     train_transform = transforms.Compose(
    #         [transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, padding=4)])

    FLAGS_Final = (0,0,0)
    if args.pipeline:
        if (args.pipeline[-1] == "augmix"):
            preprocess_aug = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(MEAN, STD)])
            FLAG_AugFinal = True
            #if len(args.pipeline) < 2:
            #    FLAG_AugFinal = False
        else:
            preprocess_aug = transforms.Compose([transforms.ToTensor()])
            FLAG_AugFinal = False

        if (args.pipeline[-1] == "prime"):
            preprocess_prime = transforms.Compose([transforms.Normalize(MEAN,STD)])
            FLAG_PrimeFinal = True
            #if len(args.pipeline) < 2:
            #    FLAG_PrimeFinal = False
        else:
            preprocess_prime = 0.0
            FLAG_PrimeFinal = False


        if (args.pipeline[-1] == "confusion"):
            preprocess_confusion = transforms.Compose([transforms.Normalize(MEAN,STD)])
            FLAG_ConfusionFinal = True
            #if len(args.pipeline) < 2:
            #    FLAG_ConfusionFinal = False
        else:
            preprocess_confusion = 0.0
            FLAG_ConfusionFinal = False

        FLAGS_Final = (FLAG_AugFinal,FLAG_PrimeFinal,FLAG_ConfusionFinal)



    #comp_list_test = [transforms.ToTensor()]
    
    if (args.dataset == "CIFAR10"):

        trainset_basic = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=gen_transform_train)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)

        #both augmix and PRIME want [crop, flip] before their augmentations
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 10
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 32, 32
            
    elif (args.dataset == "CIFAR100"):

        trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_train)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 100
        nclass=100
        nchannels = 3
        H, W = 32, 32
    elif (args.dataset == "IN100"):
        
        trainset_basic = datasets.ImageFolder(
            './Data_ImageNet/train_100',
            transform=gen_transform_train)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './Data_ImageNet/train_100',
            transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './Data_ImageNet/val_100',
            transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 100
        nclass = 100
        nchannels = 3
        H, W = 224, 224

    else:
          
        print ("Error getting dataset")



    transformDict = {}

    transformDict['basic'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4),transforms.Normalize(MEAN, STD)])
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
    transformDict['mean'] = MEAN
    transformDict['std'] = STD

    # if kwargsUser['augmix']:
    #     transformDict['aug'] = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4), transforms.AugMix()])

    protolist = []
    proto_no_norm = []

    if kwargsUser['proto_aug']:
        if "crop" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomCrop(H, padding=4))
            proto_no_norm.append(transforms.RandomCrop(H, padding=4))
        if "flip" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomHorizontalFlip(p=0.5))
            proto_no_norm.append(transforms.RandomHorizontalFlip(p=0.5))
        if "invert" in kwargsUser['proto_aug']:
            protolist.append(transforms.RandomInvert(p=0.5))
            proto_no_norm.append(transforms.RandomInvert(p=0.5))

    protolist.append(transforms.Normalize(MEAN, STD))
    #print ("transform protos")
    transformDict['proto'] = transforms.Compose(protolist)
    transformDict['proto_no_norm'] = transforms.Compose(proto_no_norm)

    if kwargsUser['prime']:
        augmentations_prime = []

        diffeo = Diffeo(
                sT=1., rT=1.,
                scut=1., rcut=1.,
                cutmin=2, cutmax=100,
                alpha=1.0, stochastic=True
            )
        augmentations_prime.append(diffeo)


        color = RandomSmoothColor(
                cut=100, T=0.01,
                freq_bandwidth=None, stochastic=True
            )
        augmentations_prime.append(color)


        filt = RandomFilter(
                kernel_size=3,
                sigma=4.0, stochastic=True
            )
        augmentations_prime.append(filt)

        prime_mod = PRIMEAugModule(augmentations_prime)

    #writer = SummaryWriter()
    #make par images
    with torch.no_grad():

        par_image_list = []

        if kwargsUser['latent_proto']:

            par_images_glob = torch.rand([kwargsUser['num_classes'],512], dtype=torch.float, device=device)

            if kwargsUser['proto_norm']:
                par_images_glob = F.normalize(par_images_glob)

        else:
            par_images_glob = torch.rand([kwargsUser['num_classes'],channels,H,W],dtype=torch.float, device=device)

            for _ in range(args.class_centers):
                par_image_list.append(par_images_glob.clone().detach())

            par_images_glob = torch.cat(par_image_list,dim=0)

            #class centers for "same label" start close together
            par_images_glob += ((0.1/255)*torch.rand([par_images_glob.size(0),channels,H,W],dtype=torch.float, device=device))
            par_images_glob.clamp_(0.0,1.0)



    ################################### DATA PIPELINE #####################################
    prime_wrapper=0.0
    confuse_wrapper=0.0

    if kwargsUser['confusion']:
        n = kwargsUser['num_classes']
        cur_confusion = [[] for i in range(n)]

    if kwargsUser['augmix'] and kwargsUser['confusion']:

        if FLAG_AugFinal:
            train_data_conf = ConfusionAugDataset(trainset, preprocess_confusion, proto_preprocess = proto_no_norm, num_classes=kwargsUser['num_classes'],
                                                  prototypes = par_images_glob, confusionHash=cur_confusion, js_loss = args.js_loss, m_ave = args.alpha_mix,
                                                  final_process=FLAG_ConfusionFinal, pipelength = kwargsUser['pipelength'], confusionMode= args.confusion_mode,
                                                  mode0rand=args.mode0rand, window=args.window, counts=args.counts) 

            train_data_aug = AugMixDataset(train_data_conf, preprocess_aug, args.js_loss, final_process=FLAG_AugFinal,pipelength = kwargsUser['pipelength'])
            cur_loader = torch.utils.data.DataLoader(train_data_aug, batch_size=args.batch_size, shuffle=True, **kwargs)
        elif FLAG_ConfusionFinal:
            train_data_aug = AugMixDataset(trainset, preprocess_aug, args.js_loss, final_process=FLAG_AugFinal,pipelength = kwargsUser['pipelength'])
          
            train_data_conf = ConfusionAugDataset(train_data_aug, preprocess_confusion, proto_preprocess = proto_no_norm, num_classes=kwargsUser['num_classes'],
                                                  prototypes = par_images_glob, confusionHash=cur_confusion, js_loss = args.js_loss, m_ave = args.alpha_mix,
                                                  final_process=FLAG_ConfusionFinal, pipelength = kwargsUser['pipelength'], confusionMode= args.confusion_mode,
                                                  mode0rand=args.mode0rand, window=args.window, counts=args.counts) 
            cur_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            print ("ERROR organizing augmix and confusion datasets")


    #DOES OLD METHOD OF MIXUP STILL WORK?
    #YES, just exclude 'confusion' in pipeline and let prime and augmix ride solo

    if kwargsUser['prime'] and kwargsUser['confusion']:

        if FLAG_PrimeFinal:

            prime_wrapper = PrimeMixWrapper(preprocess_prime, prime_mod, js_loss=args.js_loss, final_process=FLAG_PrimeFinal, pipelength = kwargsUser['pipelength'])
            train_data_conf = ConfusionAugDataset(trainset, preprocess_confusion, proto_preprocess = proto_no_norm, num_classes=kwargsUser['num_classes'],
                prototypes = par_images_glob, confusionHash=cur_confusion, js_loss = args.js_loss, m_ave = args.alpha_mix,
                                                  final_process=FLAG_ConfusionFinal, pipelength = kwargsUser['pipelength'], confusionMode= args.confusion_mode,
                                                  mode0rand=args.mode0rand, window=args.window, counts=args.counts) 
            cur_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=args.batch_size, shuffle=True, **kwargs)

        elif FLAG_ConfusionFinal:

            prime_wrapper = PrimeMixWrapper(preprocess_prime, prime_mod, js_loss=args.js_loss, final_process=FLAG_PrimeFinal, pipelength = kwargsUser['pipelength'])
            confuse_wrapper = ConfusionAugWrapper(preprocess_confusion, proto_preprocess = proto_no_norm, num_classes=kwargsUser['num_classes'],
                prototypes = par_images_glob, confusionHash=cur_confusion, js_loss = args.js_loss, m_ave = args.alpha_mix,
                                                  final_process=FLAG_ConfusionFinal, pipelength = kwargsUser['pipelength'], confusionMode=args.confusion_mode,
                                                  mode0rand=args.mode0rand, window=args.window, counts=args.counts)
            cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

        else:
            print ("ERROR organizing prime and confusion pipeline")


    #SOLO AND BASELINE CASES (SINGLE AUGMENTATIONS)

    if kwargsUser['augmix'] and kwargsUser['pipelength']==1:
        #augmix only
        train_data_aug = AugMixDataset(trainset, preprocess_aug, args.js_loss, final_process=FLAG_AugFinal)
        cur_loader = torch.utils.data.DataLoader(train_data_aug, batch_size=args.batch_size, shuffle=True, **kwargs)

    if kwargsUser['confusion'] and kwargsUser['pipelength']==1:
        train_data_conf = ConfusionAugDataset(trainset, preprocess_confusion, proto_preprocess = proto_no_norm, num_classes=kwargsUser['num_classes'],
                                              prototypes = par_images_glob, confusionHash=cur_confusion, js_loss = args.js_loss, m_ave = args.alpha_mix,
                                              final_process=FLAG_ConfusionFinal, confusionMode=args.confusion_mode, mode0rand=args.mode0rand,
                                              window=args.window, counts=args.counts) 
        cur_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=args.batch_size, shuffle=True, **kwargs)

    if kwargsUser['prime'] and kwargsUser['pipelength']==1:
        prime_wrapper = PrimeMixWrapper(preprocess_prime, prime_mod, js_loss=args.js_loss, final_process=FLAG_PrimeFinal)
        cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    if (not args.pipeline) and kwargsUser['pipelength']==0:
        print ("no algorithmic data augmentation")
        




    # init model, ResNet18() can be also used here for training
    if (args.model == "ResNet18"):
        if args.dataset in ["CIFAR10","CIFAR100"]:
            model = ResNet18(nclass = nclass, scale=args.model_scale, **kwargsUser).to(device)
            image_model = ResNet18(nclass = nclass, scale=args.model_scale,**kwargsUser).to(device)
        elif args.dataset in ["IN100"]:
            model = ResNet18IN(nclass=nclass,scale=args.model_scale, **kwargsUser).to(device)
            image_model = ResNet18IN(nclass = nclass, scale=args.model_scale, **kwargsUser).to(device)
        else:
            print ("Error matching model to dataset")
    elif (args.model == "LogNetBaseline"):
        model = LogNetBaseline(nclass=nclass, scale=1.0, **kwargsUser).to(device)
        image_model = LogNetBaseline(nclass=nclass, scale=1.0, **kwargsUser).to(device)
    elif (args.model == "PreActResNet18"):
        model = PreActResNet18(nclass = nclass,**kwargsUser).to(device)
        image_model = PreActResNet18(nclass = nclass,**kwargsUser).to(device)
    elif (args.model == "WRN16_2"):
        model = WRN16_2(nclass=nclass,**kwargsUser).to(device)
        image_model = WRN16_2(nclass = nclass,**kwargsUser).to(device)
    elif (args.model == "WRN16_4"):
        model = WRN16_4(nclass = nclass,**kwargsUser).to(device)
        image_model = WRN16_4(nclass = nclass,**kwargsUser).to(device)
    elif (args.model == "DenseNet"):
        model = densenet(num_classes = kwargsUser['num_classes']).to(device)
        image_model = densenet(num_classes = kwargsUser['num_classes']).to(device)
    elif (args.model == "ResNeXt"):
        model = resnext29(num_classes = kwargsUser['num_classes']).to(device)
        image_model = resnext29(num_classes = kwargsUser['num_classes']).to(device)
    elif (args.model == "AllConv"):
        model = AllConvNet(num_classes = kwargsUser['num_classes']).to(device)
        image_model = AllConvNet(num_classes = kwargsUser['num_classes']).to(device)
    else:
        print ("Invalid model architecture")

    image_model.multi_out =1
    image_model.eval()
    # if args.wide:
    #     model = WideResNet(**kwargsUser).to(device)
    # else:
    #     model = ResNet18(**kwargsUser).to(device)
    #     image_model = ResNet18(**kwargsUser).to(device)
    #     image_model.multi_out = 1
    #     image_model.eval()

    if (not args.image_train):
        if args.anneal in ["stairstep", "cosine"]:
            lr_i = args.lr
        elif args.anneal in ["cyclic"]:
            lr_i = 0.2
        else:
            print ("Error setting learning rate")

        print (lr_i)
        optimizer = optim.SGD(model.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        scheduler = 0.0

        print ("len(cur_loader.dataset)", len(cur_loader.dataset))
        print ("len(cur_loader)", len(cur_loader))
        
        steps_per_epoch = int(np.ceil(len(cur_loader.dataset) / args.batch_size))
        if args.anneal == "stairstep":
            pass
        elif args.anneal == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
        elif args.anneal == "cyclic":
            pct_start = 0.25
            #steps_per_epoch = 391   #50k / 128
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_i, epochs = args.epochs, steps_per_epoch = steps_per_epoch, pct_start = pct_start)
        else:
            print ("ERROR making scheduler") 
    else:
        model_pnt = torch.load('{}/model-{}-epoch{}.pt'.format(model_dir,network_string,args.restart_epoch))
        model.load_state_dict(model_pnt)
        model.eval()
        for p in model.parameters(): 
            p.requires_grad = False


    #image_model = copy.deepcopy(model)
    #image_model.eval()

    if (args.restart):
        model_pnt = torch.load('{}/model-{}-epoch{}.pt'.format(model_dir,network_string,args.restart_epoch))
        opt_pnt = torch.load('{}/opt-{}-checkpoint_epoch{}.tar'.format(model_dir,network_string,args.restart_epoch))
        model.load_state_dict(model_pnt)
        optimizer.load_state_dict(opt_pnt)


    #initialize confusionList

    confusionList = []

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        if args.anneal == "stairstep":
            adjust_learning_rate(optimizer, epoch)

        model.multi_out = 1
        par_images_glob_ref = par_images_glob.clone().detach()
        
        with open('{}/lr_hist.txt'.format(model_dir), 'a') as f:
            f.write("{0:6.8f}".format(scheduler.get_last_lr()[0]))
            f.write("\n")
        f.close()

        print (scheduler.get_last_lr()[0])
        # proximity training
        if (args.image_train):
            train_image(args, model, device, train_loader, epoch, par_images_glob, transformDict=transformDict, **kwargsUser)
        else:
            train(args, model, image_model, device, cur_loader, optimizer, epoch, par_images_glob, scheduler=scheduler, max_steps = steps_per_epoch, par_sparse=args.par_sparse, zeta=args.par_zeta, PrimeWrapper=prime_wrapper, ConfuseWrapper=confuse_wrapper, transformDict=transformDict, flags=FLAGS_Final, **kwargsUser)

        model.multi_out = 0
        
        #par_images_glob.data = par_update.data
        
        with torch.no_grad():
            par_change = torch.mean(torch.linalg.norm((par_images_glob - par_images_glob_ref).view(par_images_glob.shape[0],-1),2, dim=1))

        # evaluation on natural examples
        if (not args.image_train):
            print('================================================================')
            loss_train, acc_train = eval_train(model, device, train_loader, transformDict)
            loss_test, acc_test = eval_test(model, device, test_loader, transformDict)
            print ("parametric images mean {0:4.3f}".format(torch.mean(par_images_glob).item()))
            print ("parametric images change {0:4.8f}".format(par_change.item()))
            print('================================================================')

        if (kwargsUser['confusion'] and epoch >= args.expand_data_epoch and (((epoch-args.expand_data_epoch)%args.mix_interval)==0)):

            cur_confusion = recompute_confusion(model,device,par_images_glob, cur_confusion, num_classes=kwargsUser['num_classes'], transformDict=transformDict)
            if kwargsUser['augmix']:
                if (FLAGS_Final[2]):
                    cur_loader.dataset.active = 1
                    cur_loader.dataset.update_confusion(cur_confusion)
                else:
                    cur_loader.dataset.dataset.active = 1
                    cur_loader.dataset.dataset.update_confusion(cur_confusion)

            elif kwargsUser['prime']:
                if (FLAGS_Final[2]):
                    confuse_wrapper.active = 1
                    confuse_wrapper.update_confusion(cur_confusion)
                else:
                    cur_loader.dataset.active = 1
                    cur_loader.dataset.update_confusion(cur_confusion)
            else:
                cur_loader.dataset.active = 1
                cur_loader.dataset.update_confusion(cur_confusion)


        #just realized my customdataset class does not have augmentations

        if args.expand_data_epoch > 0 and epoch >= args.expand_data_epoch and (epoch < (args.epochs-5)):

            if (args.inter_mix and (((epoch-args.expand_data_epoch) % args.expand_interval) == 0)) or (args.mixup and (((epoch-args.expand_data_epoch)%args.mix_interval)==0)):
                #code or function call to expand training set
                print ("expanding data")
                delta_set, confusionList = expand_data(args, model, device, train_loader_basic, par_images_glob, epoch, transformDict, confusionList, **kwargsUser)
                if (len(delta_set) > 0):
                    delta_dataloader = torch.utils.data.DataLoader(delta_set, batch_size=200, shuffle=False, **kwargs)
                    xm, ym = merge_data(train_loader_basic,delta_dataloader)
                    merged_set = CustomDataSet(xm, ym, train_transform_tensor) 
                    #May need to modify this with new implementation of Augmix
                    cur_loader = torch.utils.data.DataLoader(merged_set, batch_size=args.batch_size, shuffle=True, **kwargs)
                    # FOR PRIME, we can leave cur_loader as is with the new merged set
                    if kwargsUser['augmix']:
                        train_data = AugMixDataset(merged_set, preprocess_aug, args.js_loss, final_process=FLAG_AugFinal,pipelength = kwargsUser['pipelength'])
                        cur_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)



        if (not args.image_train):
            with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
                f.write("{0:4.3f}\t{1:4.3f}\t{2:4.0f}\t{3:4.3f}\t{4:6.5f}\t{5:6.5f}\n".format(acc_train,acc_test,len(cur_loader.dataset),par_change.item(),loss_train,loss_test))
            f.close()
        #writer.add_scalar("Loss/train", loss_train, epoch)
        #writer.add_scalar("Acc/train", acc_train, epoch)
        #writer.add_scalar("Loss/test", loss_test, epoch)
        #writer.add_scalar("Acc/test", acc_test, epoch)


        # save checkpoint
        # change file name here if it needs to be
        if epoch > 99:

            if (not args.image_train):
                torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-{}-epoch{}.pt'.format(network_string,epoch)))
                torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}-checkpoint_epoch{}.tar'.format(network_string,epoch)))
            #writer.flush()
            torch.save(par_images_glob, os.path.join(model_dir,'parametric_images_lyr_{}_pool_{}_epoch{}.pt'.format(args.proto_layer,args.proto_pool,epoch)))


if __name__ == '__main__':
    main()

