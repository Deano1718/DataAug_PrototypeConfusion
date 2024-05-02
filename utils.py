
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#import smoothers


def get_split_correct_points(dataloader, model, process='', **kwargs):
    
    correct_lists = []
    correct_lab_lists = []
    
    for i in range(10):
        correct_lists.append([])
        correct_lab_lists.append([])
    
    with torch.no_grad(): # tell Pytorch not to build graph in this section
        for batch_idx, data in enumerate(dataloader):

            X, Y = data[0].to(DEVICE), data[1].to(DEVICE)

            if process:
                X = process(X,**kwargs)

            Z = model(X)

            #make prediction, do not need softmax
            Yp = Z.data.max(dim=1)[1]  # get the index of the max for each batch sample
                                       # Z.data.max(dim=1) returns two tensors, [0] is values, [1] is indices

            filtered_data = X[Yp==Y].clone().detach()
            filtered_labels = Y[Yp==Y].clone().detach()

            
            for j in range(10):
                correct_lists[j].append(filtered_data[filtered_labels==j].clone().detach())
                correct_lab_lists[j].append(filtered_labels[filtered_labels==j].clone().detach())

            
        for L in range(len(correct_lists)):
            correct_lists[L] = torch.cat(correct_lists[L])
            correct_lab_lists[L] = torch.cat(correct_lab_lists[L])
                

    return correct_lists, correct_lab_lists



def get_split_points(dataloader,nclass,**kwargs):

    class_lists = []
    class_lab_lists = []
    
    for i in range(nclass):
        class_lists.append([])
        class_lab_lists.append([])
    
    with torch.no_grad(): # tell Pytorch not to build graph in this section
        for batch_idx, data in enumerate(dataloader):

            X, Y = data[0].cpu(), data[1].cpu()

            
            for j in range(nclass):
                class_lists[j].append(X[Y==j].clone().detach())
                class_lab_lists[j].append(Y[Y==j].clone().detach())

            
    for L in range(len(class_lists)):
        class_lists[L] = torch.cat(class_lists[L])
        class_lab_lists[L] = torch.cat(class_lab_lists[L])
                

    return class_lists, class_lab_lists

def compute_variance(device, model, dataloader, nclass, transformDict, norm=0):
    model.multi_out=1

    print ("splitting data")
    xl, yl = get_split_points(dataloader,nclass)

    ave_variances = []
    max_variances = []
    mean_std_ratios = []
    maxmean_std_ratios = []

    for lbl in range(nclass):
        class_set = CustomDataSet(xl[lbl],yl[lbl])
        #class_set_test = CustomDataSet(xl_test[lbl], yl_test[lb])
        
        img_loader = DataLoader(class_set, batch_size=100, shuffle=False)
        #img_loader_test = DataLoader(class_set_test, batch_size=100, shuffle=False)
        
        with torch.no_grad(): # tell Pytorch not to build graph in this section                                                                                                                                        
            feature_list = []

            #TRAINING EXAMPLES                                         
            for batch_idx, data in enumerate(img_loader):

                X, Y = data[0].to(device), data[1].cpu()
                bs = X.size(0)

                X = transformDict['norm'](X)

                L2, logits = model(X)
                if norm:
                    L2 = F.normalize(L2)
                L2 = torch.abs(L2).cpu()
                logits = logits.cpu()
                feature_list.append(L2.clone().cpu())

            feature_list = torch.cat(feature_list,dim=0)   # [num_examples, 512]                                                                                                                                       
            feature_stds, feature_means = torch.std_mean(feature_list,unbiased=False,dim=0)   #[512]                                                                                                                   
            feature_vars = feature_stds**2
            sorted_means, sorted_means_idx = torch.sort(feature_means)
            ave_variances.append(torch.mean(feature_vars).clone())
            max_variances.append(torch.max(feature_vars).clone())
            cur_ratios = torch.nan_to_num(feature_means/feature_stds)
            cur_ratios_sorted = cur_ratios[sorted_means_idx]
            median_ratio = torch.median(cur_ratios_sorted[int(len(cur_ratios_sorted)/2):])
            #mean_std_ratios.append(torch.nan_to_num(feature_means/feature_stds).clone())
            mean_std_ratios.append(median_ratio.clone())
            maxmean_std_ratios.append(cur_ratios_sorted[-1])

    return torch.mean(torch.stack(ave_variances,dim=0)), torch.mean(torch.stack(max_variances,dim=0)), torch.mean(torch.stack(mean_std_ratios, dim=0)), torch.mean(torch.stack(maxmean_std_ratios, dim=0))


def get_split_points_weighted(dataloader,**kwargs):

    class_lists = []
    class_lab_lists = []
    weight_lists = []
    x_id_lists = []
    
    for i in range(kwargs['num_classes']):
        class_lists.append([])
        class_lab_lists.append([])
        weight_lists.append([])
        x_idx_lists.append([])
    
    with torch.no_grad(): # tell Pytorch not to build graph in this section
        for batch_idx, data in enumerate(dataloader):

            X, Y, W, x_id = data[0].cpu(), data[1].cpu(), data[2].cpu(), data[3].cpu()

            
            for j in range(kwargs['num_classes']):
                class_lists[j].append(X[Y==j].clone().detach())
                class_lab_lists[j].append(Y[Y==j].clone().detach())
                weight_lists[j].append(W[Y==j].clone().detach())
                x_id_lists[j].append(x_id[Y==j].clone().detach())

            
    for L in range(len(class_lists)):
        class_lists[L] = torch.cat(class_lists[L])
        class_lab_lists[L] = torch.cat(class_lab_lists[L])
        weight_lists[L] = torch.cat(weight_lists[L])
        x_id_lists[L] = torch.cat(x_id_lists[L])
                

    return class_lists, class_lab_lists, weight_lists, x_id_lists


def merge_data(dataset_loader1, dataset_loader2):


    xdat = []
    ydat = []

    print ('Merging data')


    for batch_idx, data in enumerate(dataset_loader1):

        X, Y = data[0].cpu(), data[1].cpu()

        xdat.append(X.clone())
        ydat.append(Y.clone())


    for batch_idx, data in enumerate(dataset_loader2):

        X, Y = data[0].cpu(), data[1].cpu()

        xdat.append(X.clone())
        ydat.append(Y.clone())

        # if (batch_idx % 10 == 0):
        #     print ('batch', batch_idx, 'complete')

    xdatcat = torch.cat(xdat,dim=0)
    ydatcat = torch.cat(ydat,dim=0)
        
    return xdatcat, ydatcat


class SwapDataSet(Dataset):
    def __init__(self, dataset, transform=[]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        with torch.no_grad():
            idx = torch.randperm(3)
            x = x[idx]
            #x = x[idx,:,:]
        
        if self.transform:
            return self.transform(x), y
        else:
            return x, y

class CustomDataSetWTransform(Dataset):
    def __init__(self, dataset, transform=[]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        if self.transform:
            x_transform = self.transform(x)
            return x_transform, y
        else:
            return x, y

class CustomDataSet(Dataset):   

    def __init__(self, Xdata, Ydata, transform=[]):
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.transform = transform
        
    def __len__(self):
        return len(self.Xdata)   
    
    def __getitem__(self, index):

        if self.transform:
            return self.transform(self.Xdata[index]), self.Ydata[index]
        else:
            return self.Xdata[index], self.Ydata[index]

class CustomDataSetWNoise(Dataset):

    def __init__(self, dataset, transform=[]):
        #self.Xdata = Xdata
        #self.Ydata = Ydata
        self.dataset = dataset
        self.transform = transform
        
        self.global_noise_data = 0.015*((torch.rand([len(self.dataset),3,32,32],dtype=torch.float) - 0.5)/0.5)
        print (self.global_noise_data.shape)

    def update_noise(self, new_noise, indices):
        self.global_noise_data[indices].data = new_noise.data
        
    def __len__(self):
        return len(self.dataset)   
    
    def __getitem__(self, index):

        x,y = self.dataset[index]

        if self.transform:
            return self.transform(x), y, self.global_noise_data[index], index
        else:
            return x, y, self.global_noise_data[index], index



class CustomDataSetID(Dataset):   

    def __init__(self, Xdata, Ydata, transform=[]):
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.transform = transform
        
    def __len__(self):
        return len(self.Xdata)   
    
    def __getitem__(self, index):

        if self.transform:
            return self.transform(self.Xdata[index]), self.Ydata[index], index
        else:
            return self.Xdata[index], self.Ydata[index], index


class CustomDataSetWeightedDS(Dataset):

    def __init__(self, dataset, active_weights=1, transform=[]):
        self.dataset = dataset
        
        #self.dataset = dataset                                                                                                                                                                                                        
        self.active_weights = active_weights
        self.transform = transform
        #self.Wdata = Wdata                                                                                                                                                                                                            

        with torch.no_grad():
            if self.active_weights:
                self.Wdata = torch.ones((len(self.dataset)), dtype=torch.float)
            else:
                self.Wdata = torch.zeros((len(self.dataset)), dtype=torch.float)

    def renorm_weights(self):
        with torch.no_grad():
            total_weight = torch.sum(self.Wdata)
            self.Wdata /= total_weight
            self.Wdata *= len(self.dataset)

    def change_weights(self, indices, factor):
        with torch.no_grad():
            self.Wdata[indices] *= factor

    def report_min_max(self):

        with torch.no_grad():
            #sum_change = torch.sum(torch.ne(self.Wdata, torch.ones_like(self.Wdata)))                                                                                                                                                 
            w_std, w_mean = torch.std_mean(self.Wdata,unbiased=False)
            return (torch.min(self.Wdata), torch.max(self.Wdata), w_std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        #print ("should transform")

        x,y = self.dataset[index]
        
        if self.transform:
            return self.transform(x), y, self.Wdata[index], index
        else:
            return x, y, self.Wdata[index], index


class CustomDataSetWeighted(Dataset):   

    def __init__(self, data, labels, active_weights=1, transform=[]):
        self.Xdata = data
        self.Ydata = labels
        #self.dataset = dataset
        self.active_weights = active_weights
        self.transform = transform
        #self.Wdata = Wdata

        with torch.no_grad():
            if self.active_weights:
                self.Wdata = torch.ones_like(self.Ydata, dtype=torch.float)
            else:
                self.Wdata = torch.zeros_like(self.Ydata, dtype=torch.float)

    def renorm_weights(self):
        with torch.no_grad():
            total_weight = torch.sum(self.Wdata)
            self.Wdata /= total_weight
            self.Wdata *= len(self.Ydata)

    def change_weights(self, indices, factor):
        with torch.no_grad():
            self.Wdata[indices] *= factor

    def report_min_max(self):

        with torch.no_grad():
            #sum_change = torch.sum(torch.ne(self.Wdata, torch.ones_like(self.Wdata)))
            w_std, w_mean = torch.std_mean(self.Wdata,unbiased=False)
            return (torch.min(self.Wdata), torch.max(self.Wdata), w_std)
        
    def __len__(self):
        return len(self.Ydata)   
    
    def __getitem__(self, index):
        #print ("should transform")
        if self.transform:
            return self.transform(self.Xdata[index]), self.Ydata[index], self.Wdata[index], index
        else:
            return self.Xdata[index], self.Ydata[index], self.Wdata[index], index

class CustomDataSetWeightedTest(Dataset):

    def __init__(self, ds, transform=[]):

        self.dataset = ds
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        x, y = self.dataset[index]

        if self.transform:
            return self.transform(x), y, 1, index
        else:
            return x, y, 1, index



# class CustomDataSetWeighted(Dataset):   

#     def __init__(self, dataset, Wdata, transform=[]):
#         #self.Xdata = Xdata
#         #self.Ydata = Ydata
#         self.dataset = dataset
#         self.transform = transform
#         self.Wdata = Wdata

#         with torch.no_grad():
#             self.Wdata = torch.ones_like(self.Ydata)

#     def renorm_weights(self):
#         with torch.no_grad():
#             total_weight = torch.sum(self.Wdata)
#             self.Wdata /= total_weight

#     def change_weights(self, indices, factor):
#         with torch.no_grad():
#             self.Wdata[indices] *= factor
        
#     def __len__(self):
#         return len(self.dataset)   
    
#     def __getitem__(self, index):

#         x, y = self.dataset[index]

#         if self.transform:
#             return self.transform(x), y, self.Wdata[index], index
#         else:
#             return self.Xdata[index], self.Ydata[index], self.Wdata[index], index


def get_data_stats(loader, device):

    mean = 0.0
    std = 0.0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            mean += torch.sum(torch.mean(data.view(data.shape[0],-1),dim=1),dim=0)
        mean /= len(loader.dataset)

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            std += torch.sum(torch.mean( ((data - mean)**2).view(data.shape[0],-1),dim=1),dim=0)

        std /= (len(loader.dataset))

        std = std**0.5


    return mean, std

def boost_prototypes(args, model, device, dloader, targets, transformDict={}, weight_factor=2.0, sigma_factor=3.0, **kwargs):

    model.eval()
    model.multi_out = 1
    #par_tens = par_tens.clone().cpu()
    num_classes = kwargs['num_classes']

    #track proximity to prototypes on first go round
    prox_data = []
    prox_means = []
    prox_devs = []

    for _ in range(num_classes):
        prox_data.append([])


    # if k != 0:
    #     #compute mask
    #     with torch.no_grad():
    #         # x_0 = torch.zeros_like(x)
    #         # x_1 = torch.ones_like(x)
    #         # idx_x = torch.topk(x, np.absolute(self.k), dim=1, largest = np.sign(self.k))[1]
    #         # mask_x = x_0.scatter_(1, idx_x, x_1)

    #         proto_0 = torch.zeros_like(targets)
    #         proto_1 = torch.ones_like(targets)
    #         idx_targets = torch.topk(targets, k, dim=1, largest = self.largest)[1]
    #         mask_targets = proto_0.scatter_(dim=1, idx_targets, proto_1)

    #         #recompute this depending on class being addressed
    #         #mask_x = mask_targets[labels]

    #     targets_ = targets*mask_targets
    #     #x_ = x*mask_x
    # else:
    #     targets_ = targets
    #     x_ = x

    with torch.no_grad():

        #compute distance to class prototype (all latent features) for each example
        for batch_idx, data in enumerate(dloader):

            x, y, w, x_id = data[0].to(device), data[1].to(device), data[2].cpu(), data[3].cpu()

            #print (x.shape)
            #print (model(transformDict['norm'](x)))
            
            x_p, z = model(transformDict['norm'](x))

            #x_p = x_p.cpu()
            #xp gpu
            #targets gpu
            #y on cpu

            dists = torch.linalg.norm(x_p - targets[y], ord=2, dim=1)

            for j in range(kwargs['num_classes']):

                prox_data[j].append(dists[y==j].clone().detach())

        #compute statistics and run through again to update weights
        #do them one at a time and store mean/dev in separate lists
        for l in prox_data:
            tensl = torch.cat(l)
            #print (tensl.shape)
            prox_std, prox_mean = torch.std_mean(tensl,unbiased=False)
            prox_means.append(prox_mean.clone())
            prox_devs.append(prox_std.clone())

        prox_means = torch.stack(prox_means, dim=0)
        prox_devs = torch.stack(prox_devs, dim=0)

        print (prox_means.shape)


        for batch_idx, data in enumerate(dloader):

            x, y, w, x_id = data[0].to(device), data[1].to(device), data[2].cpu(), data[3].cpu()

            x_p, z = model(transformDict['norm'](x))

            #x_p = x_p.cpu()

            dists = torch.linalg.norm(x_p - targets[y], ord=2, dim=1)

            means_cur = prox_means[y].clone()
            devs_cur = prox_devs[y].clone()

            modify = x_id[dists > (means_cur + sigma_factor*devs_cur)].clone()

            dloader.dataset.dataset.change_weights(indices=modify, factor=weight_factor)

            dloader.dataset.dataset.renorm_weights()


# A Python3 program for
# Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix
# representation of the graph
 
 
class MSTGraph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    # A utility function to print
    # the constructed MST stored in parent[]
    def printMST(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):
 
        # Initialize min value
        minval = 1.e12
 
        for v in range(self.V):
            if key[v] < minval and mstSet[v] == False:
                minval = key[v]
                min_index = v
 
        return min_index
 
    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):
 
        # Key values used to pick minimum weight edge in cut
        key = [1.e8] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V
 
        parent[0] = -1  # First node is always the root of
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)
 
            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
 
                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
 
        #self.printMST(parent)
        return parent
 
 
# # Driver's code
# if __name__ == '__main__':
#     g = Graph(5)
#     g.graph = [[0, 2, 0, 6, 0],
#                [2, 0, 3, 8, 5],
#                [0, 3, 0, 0, 7],
#                [6, 8, 0, 0, 9],
#                [0, 5, 7, 9, 0]]
 
#     g.primMST()
 
 
# Contributed by Divyanshu Mehta

