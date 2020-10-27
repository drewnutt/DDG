#!/usr/bin/env python3
# coding: utf-8

# Train to differences from linear model


import pyanitools as pya
import molgrid
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os, glob
import matplotlib.pyplot as plt
import wandb
import argparse, pickle



parser = argparse.ArgumentParser(description='Progressively train on ANI data (which is in current directory)')

parser.add_argument("--maxepoch",default=100,type=int,help="Number of epochs before moving on")
parser.add_argument("--stop",default=20000, type=int, help="Number of iterations without improvement before moving on")
parser.add_argument("--lr",default=0.001,type=float, help="Initial learning rate")
parser.add_argument("--resolution",default=0.25,type=float, help="Grid resolution")
parser.add_argument("--clip",default=10,type=float, help="Gradient clipping")
parser.add_argument("--solver",default="adam",choices=('adam','sgd'),type=str, help="solver to use (adam|sgd)")
parser.add_argument("--pickle",default="traintest.pickle",type=str)
parser.add_argument("--num_modules",default=5,type=int,help="number of convolutional modules")
parser.add_argument("--module_depth",default=1,type=int,help="number of layers in module")
parser.add_argument("--module_connect",default="straight",choices=('straight','dense','residual'),type=str, help="how module is connected")
parser.add_argument("--module_kernel_size",default=3,type=int,help="kernel size of module")
parser.add_argument("--module_filters",default=64,type=int,help="number of filters in each module")
parser.add_argument("--filter_factor",default=2,type=float,help="set filters to this raised to the current module index")
parser.add_argument("--activation_function",default="elu",choices=('elu','relu','sigmoid'),help='activation function')
parser.add_argument("--hidden_size",default=0,type=int,help='size of hidden layer, zero means none')
parser.add_argument("--pool_type",default="max",choices=('max','ave'),help='type of pool to use between modules')

args = parser.parse_args()

typemap = {'H': 0, 'C': 1, 'N': 2, 'O': 3} #type indices
typeradii = [1.0, 1.6, 1.5, 1.4] #not really sure how sensitive the model is to radii


#load data
(train, test) = pickle.load(open(args.pickle,'rb'))

def load_examples(T):
  examples = []
  for coord, types, energy, diff in T:
    radii = np.array([typeradii[int(index)] for index in types], dtype=np.float32)
    c = molgrid.CoordinateSet(coord, types, radii,4)
    ex = molgrid.Example()
    ex.coord_sets.append(c)
    ex.labels.append(diff)        
    examples.append(ex)
  return examples
  
examples = load_examples(train)
valexamples = load_examples(test)
  

class View(nn.Module):
    def __init__(self, shape):        
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(*self.shape)
        
#this is Daniela's model
class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.modules = []
        self.residuals = []
        nchannels = dims[0] 
        dim = dims[1]
        ksize = args.module_kernel_size
        pad = ksize//2
        fmult = 1
        func = F.elu
        if args.activation_function == 'relu':
            func = F.relu
        elif args.activation_function == 'sigmoid':
            func = F.sigmoid
            
        inmultincr = 0
        if args.module_connect == 'dense':
            inmultincr = 1
            
        for m in range(args.num_modules):
            module = []          
            inmult = 1
            filters = int(args.module_filters*fmult  )
            startchannels = nchannels
            for i in range(args.module_depth):
                conv = nn.Conv3d(nchannels*inmult, filters, kernel_size=ksize, padding=pad)
                inmult += inmultincr
                self.add_module('conv_%d_%d'%(m,i), conv)
                module.append(conv)
                module.append(func)
                nchannels = filters
            
            if args.module_connect == 'residual':
                #create a 1x1x1 convolution to match input filters to output
                conv = nn.Conv3d(startchannels, nchannels, kernel_size=1, padding=0)
                self.add_module('resconv_%d'%m,conv)
                self.residuals.append(conv)
            #don't pool on last module
            if m < args.num_modules-1:
                pool = nn.MaxPool3d(2)
                self.add_module('pool_%d'%m,pool)
                module.append(pool)
                dim /= 2
            self.modules.append(module)
            fmult *= args.filter_factor
            
        last_size = int(dim**3 * filters)
        lastmod = []
        lastmod.append(View((-1,last_size)))
        
        if args.hidden_size > 0:
            fc = nn.Linear(last_size, args.hidden_size)
            self.add_module('hidden',fc)
            lastmod.append(fc)
            lastmod.append(func)
            last_size = args.hidden_size
            
        fc = nn.Linear(last_size, 1)
        self.add_module('fc',fc)
        lastmod.append(fc)
        lastmod.append(nn.Flatten())
        self.modules.append(lastmod)
            

    def forward(self, x):
        isdense = False
        isres = False
        if args.module_connect == 'dense':
            isdense = True
        if args.module_connect == 'residual':
            isres = True
                        
        for (m,module) in enumerate(self.modules):
            prevconvs = []
            if isres and len(self.residuals) > m:
                #apply convolution
                passthrough = self.residuals[m](x)
            else:
                isres = False
            for (l,layer) in enumerate(module):
                if isinstance(layer, nn.Conv3d) and isdense:
                    if prevconvs:
                        #concate along channels
                        x = torch.cat((x,*prevconvs),1)
                if isres and l == len(module)-1:
                    #at last relu, do addition before
                    x = x + passthrough

                x = layer(x)
                
                if isinstance(layer, nn.Conv3d) and isdense:
                    prevconvs.append(x) #save for later

        return x

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)



gmaker = molgrid.GridMaker(resolution=args.resolution, dimension = 16-args.resolution)
batch_size = 25
dims = gmaker.grid_dimensions(4) # 4 types
tensor_shape = (batch_size,)+dims  #shape of batched input


#allocate tensors
input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
labels = torch.zeros(batch_size, dtype=torch.float32, device='cuda')


TRAIL = 100

def train_strata(strata, model, optimizer, losses, maxepoch, stop=20000):
    bestloss = 100000 #best trailing average loss we've seen so far in this strata
    bestindex = len(losses) #position    
    for _ in range(maxepoch):  #do at most MAXEPOCH epochs, but should bail earlier
        np.random.shuffle(strata)
        for pos in range(0,len(strata),batch_size):
            batch = strata[pos:pos+batch_size]
            if len(batch) < batch_size: #wrap last batch
                batch += strata[:batch_size-len(batch)]
            batch = molgrid.ExampleVec(batch)
            batch.extract_label(0,labels) # extract first label (there is only one in this case)

            gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
            output = model(input_tensor) #run model
            loss = F.smooth_l1_loss(output.flatten(),labels.flatten())
            loss.backward()
            
            if args.clip > 0:
              nn.utils.clip_grad_norm_(model.parameters(),args.clip)

            optimizer.step()
            losses.append(float(loss))
            trailing = np.mean(losses[-TRAIL:])
            
            if trailing < bestloss:
                bestloss = trailing
                bestindex = len(losses)
            wandb.log({'loss': float(loss),'trailing':trailing,'bestloss':bestloss,'stratasize':len(strata),'lr':optimizer.param_groups[0]['lr']})
            
            if len(losses)-bestindex > stop:
                return True # "converged"
    return False


def test_strata(valexamples, model):
    with torch.no_grad():
        model.eval()
        results = []
        labels = []
        labelvec = torch.zeros(batch_size, dtype=torch.float32, device='cuda')
        for pos in range(0,len(valexamples),batch_size):
            batch = valexamples[pos:pos+batch_size]
            if len(batch) < batch_size: #wrap last batch
                batch += valexamples[:batch_size-len(batch)]
            batch = molgrid.ExampleVec(batch)
            batch.extract_label(0,labelvec) # extract first label (there is only one in this case)

            gmaker.forward(batch, input_tensor, 2, random_rotation=True)  #create grid; randomly translate/rotate molecule
            output = model(input_tensor)   
            results.append(output.detach().cpu().numpy())
            labels.append(labelvec.detach().cpu().numpy())
            
        results = np.array(results).flatten()
        labels = np.array(labels).flatten()
        valrmse = np.sqrt(np.mean((results - labels)**2))
        valame = np.mean(np.abs(results-labels))
        print("Validation",valrmse,valame)
        wandb.log({'valrmse': valrmse,'valame':valame})
        wandb.log({'valpred':results,'valtrue':labels})
        
          
wandb.init(project="anidiff", config=args)




losses = []
model = Net(dims).to('cuda')
model.apply(weights_init)

if args.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


wandb.watch(model)

test_strata(valexamples, model)
                
#train on full training set, start stepping the learning rate
for i in range(3):
    train_strata(examples, model, optimizer, losses, args.maxepoch, args.stop)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model_refine%d.pt'%i))
    scheduler.step()
    test_strata(valexamples, model)




