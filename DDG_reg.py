import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os
import wandb
import argparse

class Net(nn.Module):
    def __init__(self, dims):
            super(Net, self).__init__()
            self.pool0 = nn.MaxPool3d(2)
            self.conv1 = nn.Conv3d(dims[0], 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool3d(2)
            self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool3d(2)
            self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

            self.last_layer_size = dims[1]//8 * dims[2]//8 * dims[3]//8 * 128
            self.fc1 = nn.Linear(self.last_layer_size, 1)

    def forward(self, x):
            x = self.pool0(x)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.last_layer_size)
            x = self.fc1(x)
            return x

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)

def calc_R2(pred,actual):
	a_mean = torch.mean(actual)
	sst,ssr = 0.0,0.0
	sst = actual.sub(a_mean)
	ssr = actual.sub(pred)
	t = (sst.pow(2)).sum().item()
	r = (ssr.pow(2)).sum().item()

	return 1-(r/t)

def calc_RMSD(pred,actual):
    diff = pred.sub(actual)
    return torch.sqrt((diff.pow(2)).sum()/len(pred)).item()

def train(args, model, traine, optimizer, epoch):
    model.train()
    train_loss = 0
    r2,rmsd = 0.0,0.0
    total = 0
    for _ in range(1708):
        batch = traine.next_batch(args.batch_size)
        gmaker.forward(batch, input_tensor,random_translation=2.0, random_rotation=True) 
        batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels,1).float().to('cuda')
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.mse_loss(output,labels)
        r2 += calc_R2(output,labels)
        rmsd += calc_RMSD(output,labels)
        total += config.batch_size
        train_loss += loss
        loss.backward()
        optimizer.step()
    return train_loss/(args.batch_size*1708), (r2/total), (rmsd/total)

def test(args, model, teste):
    model.eval()
    test_loss = 0
    tot_r2,rmsd = 0.0, 0.0
    total = 0

    with torch.no_grad():
        for _ in range(443):	
            batch = teste.next_batch(args.batch_size)
            gmaker.forward(batch, input_tensor, 0, random_rotation=False) 
            batch.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels,1).float().to('cuda')
            output = model(input_tensor)
            loss = F.mse_loss(output,labels)
            test_loss += loss
            total += args.batch_size
            tot_r2 += calc_R2(output, labels)
            rmsd += calc_RMSD(output,labels)
    return test_loss/(443*args.batch_size), tot_r2/(total), rmsd/total

parser = argparse.ArgumentParser()
parser.add_argument('--ligtr', required=True, help='location of training ligand cache file input')
parser.add_argument('--rectr', required=True, help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=True, help='location of training information')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information')
parser.add_argument('--weightdecay','-wd', default=0.0,type=float, help='weight decay for optimizer')
args = parser.parse_args()

wandb.init(entity='andmcnutt', project='DDG_model')

config = wandb.config
config.batch_size = 42
config.lr = 0.01
config.momentum = 0.9
config.threshold=0.5
config.epochs= 100
config.weight_decay=args.weightdecay

print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr,recmolcache=args.rectr, balanced=True,shuffle=True)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte, balanced=True,shuffle=True)
teste.populate(args.testfile)

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(42)
tensor_shape = (config.batch_size,)+dims

model = Net(dims).to('cuda')
model.apply(weights_init)

optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,weight_decay=config.weight_decay)

input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(config.batch_size, dtype=torch.float32)
wandb.watch(model,log='all')

for epoch in range(config.epochs):
    tr_loss, tr_r2, tr_rmsd =  train(config, model, traine, optimizer, epoch)
    tt_loss, tt_r2, tt_rmsd = test(config, model, teste)
    wandb.log({
            "Avg Train Loss": tr_loss,
            "Train R^2": tr_r2,
            "Train RMSD": tr_rmsd,
            "Avg Test Loss": tt_loss,
            "Test RMSD": tt_rmsd,
            "Test R^2": tt_r2})

torch.save(model.state_dict(), "model_reg.h5")
wandb.save('model_reg.h5')
