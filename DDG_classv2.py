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
import matplotlib.pyplot as plt

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
		self.out_act = nn.Sigmoid()

	def forward(self, x):
		x = self.pool0(x)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = F.relu(self.conv3(x))
		x = x.view(-1, self.last_layer_size)
		x = self.fc1(x)
		x = self.out_act(x)
		return x

def weights_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight.data)

def confusion(output, label, tp, fp, tn, fn):
	conf_matrix = torch.zeros(2,2)
	for out, lab in zip(output,label):
		conf_matrix[int(out),int(lab)] += 1
	tp += conf_matrix[0,0]
	fp += conf_matrix[0,1]
	fn += conf_matrix[1,0]
	tn += conf_matrix[1,1]

	return tp,fp,tn,fn

def get_AUC(pred,label):
        P = label.sum()
        N = len(label)-P
        indexes = list(range(len(pred)))
        indexes.sort(key=pred.__getitem__)
        FP, TP, fp, tp, area = 0, 0, 0, 0, 0
        prev = -float('inf')
        for i in indexes:
                if pred[i] != prev:
                        area += trap_area(FP,fp,TP,tp)
                        prev = pred[i]
                        fp = FP
                        tp = TP
                if label[i] == 1:
                        TP += 1
                else:
                        FP += 1
        area += trap_area(N,fp,N,tp)

        return 1-(area/(P*N))

def trap_area(FP,fp,TP,tp):
	return abs(FP-fp) * (TP+tp)/2

def get_labels(ll_out, threshold):
        pred_label = ll_out >= threshold
        return pred_label

def train(args, model, traine, optimizer, epoch, size):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    output_dist = []
    for _ in range(size[0]):
        batch = traine.next_batch(args.batch_size)
        gmaker.forward(batch, input_tensor,random_translation=2.0, random_rotation=True) 
        batch.extract_label(0, float_labels)
        labels = torch.unsqueeze(float_labels,1).float().to('cuda')
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output,labels)
        pred_label = get_labels(output, args.threshold)
        correct += (pred_label == labels).sum().item()
        total += config.batch_size
        train_loss += loss
        loss.backward()
        optimizer.step()
        output_dist += output.flatten().tolist()
    if size[1] != 0: #make sure go through every single training example
        batch = traine.next_batch(size[1])
        gmaker.forward(batch, tr_l_tensor,random_translation=2.0, random_rotation=True) 
        batch.extract_label(0, tr_l_labels)
        labels = torch.unsqueeze(tr_l_labels,1).float().to('cuda')
        optimizer.zero_grad()
        output = model(tr_l_tensor)
        loss = criterion(output,labels)
        pred_label = get_labels(output, args.threshold)
        correct += (pred_label == labels).sum().item()
        total += size[1]
        train_loss += loss
        loss.backward()
        optimizer.step()
        output_dist += output.flatten().tolist()

    return train_loss/(size[2]), (correct/total), output_dist

def test(args, model, teste, size):
    model.eval()
    test_loss = 0
    correct = 0
    tp,fp,tn,fn = 0,0,0,0
    tot_auc = 0


    output_dist = []
    with torch.no_grad():
        for _ in range(size[0]):	
            batch = teste.next_batch(args.batch_size)
            gmaker.forward(batch, input_tensor, 0, random_rotation=False) 
            batch.extract_label(0, float_labels)
            labels = torch.unsqueeze(float_labels,1).float().to('cuda')

            output = model(input_tensor)
            loss = criterion(output,labels)
            pred_label= get_labels(output, args.threshold)
            correct += (pred_label == labels).sum().item()
            tp,fp,tn,fn = confusion(pred_label,labels,tp,fp,tn,fn)
            test_loss += loss
            tot_auc += get_AUC(output, labels)
            output_dist += output.flatten().tolist()

        if size[1] != 0:
            batch = teste.next_batch(size[1])
            gmaker.forward(batch, tt_l_tensor, 0, random_rotation=False) 
            batch.extract_label(0, tt_l_labels)
            labels = torch.unsqueeze(tt_l_labels,1).float().to('cuda')
            output = model(tt_l_tensor)
            loss = criterion(output,labels)
            pred_label = get_labels(output, args.threshold)
            correct += (pred_label == labels).sum().item()
            tp,fp,tn,fn = confusion(pred_label,labels,tp,fp,tn,fn)
            test_loss += loss
            tot_auc += get_AUC(output, labels)
            output_dist += output.flatten().tolist()

    return test_loss/(size[2]), (tp+tn)/(tp+fp+tn+fn), (tp)/(tp+fp), (tp)/(tp+fn), tot_auc/(size[0]+int(size[1]>0)), output_dist

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
config.batch_size = 64
config.lr = 0.01
config.momentum = 0.9
config.threshold=0.5
config.epochs= 200
config.weight_decay=args.weightdecay

print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr,recmolcache=args.rectr, balanced=True,shuffle=True)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte, balanced=True,shuffle=True)
teste.populate(args.testfile)

trsize = traine.size()
tssize = teste.size()
one_e_tr = int(trsize/config.batch_size)
leftover_tr = trsize - one_e_tr * config.batch_size
one_e_tt = int(tssize/config.batch_size)
leftover_tt = tssize - one_e_tt * config.batch_size

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(42)
tensor_shape = (config.batch_size,)+dims
tr_nums=(one_e_tr,leftover_tr,trsize)
tt_nums=(one_e_tt,leftover_tt, tssize)

model = Net(dims).to('cuda')
model.apply(weights_init)

optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,weight_decay=config.weight_decay)
criterion = nn.BCELoss()

input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(config.batch_size, dtype=torch.float32)

if leftover_tr != 0:
    tr_leftover_shape = (leftover_tr,) + dims
    tr_l_tensor = torch.zeros(tr_leftover_shape, dtype=torch.float32, device='cuda')
    tr_l_labels = torch.zeros(leftover_tr, dtype=torch.float32)
if leftover_tt != 0:
    tt_leftover_shape = (leftover_tt,) + dims
    tt_l_tensor = torch.zeros(tt_leftover_shape, dtype=torch.float32, device='cuda')
    tt_l_labels = torch.zeros(leftover_tt, dtype=torch.float32)

wandb.watch(model,log='all')

for epoch in range(config.epochs):
	tr_loss, tr_acc, out_dist =  train(config, model, traine, optimizer, epoch,tr_nums)
	tt_loss, tt_acc, tt_pr, tt_rec, auc, out_d = test(config, model, teste, tt_nums)
	
	if epoch%10 == 0:
	    plt.hist(out_dist,bins=10)
	    wandb.log({"output distribution Train": [wandb.Image(plt, caption="output distribution of network")]}, commit=False)
	    plt.clf()
	    plt.hist(out_d,bins=10)
	    wandb.log({"output distribution Test": [wandb.Image(plt, caption="output distribution of model")]}, commit=False)
	wandb.log({
		"Avg Train Loss": tr_loss,
		"Train Accuracy": tr_acc,
		"Avg Test Loss": tt_loss,
		"Test Accuracy": tt_acc,
		"Test Precision": tt_pr,
		"Test Recall": tt_rec,
		"AUC": auc})

	if tr_acc == 1.00:
		print('Reached 100% train accuracy')
		break

torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
