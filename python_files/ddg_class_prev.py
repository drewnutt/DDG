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
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--ligtr', required=True, help='location of training ligand cache file input')
parser.add_argument('--rectr', required=True, help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=True, help='location of training information, this must have a group indicator')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information, this must have a group indicator')
parser.add_argument('--weightdecay','-wd', default=0.0,type=float, help='weight decay for optimizer')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="sgd", choices=('adam','sgd'), type=str, help="solver to use")
parser.add_argument('--module_depth',default=1,type=int,help="number of layers in module")
parser.add_argument('--num_modules',default=3,type=int,help="number of convolutional modules")
parser.add_argument('--module_kernel_size',default=3, type=int,help="kernel size of module")
parser.add_argument('--module_connect',default='straight',choices=('straight','dense','residual'),type=str, help='how module is connected')
parser.add_argument('--module_filters',default=64,type=int,help="number of filters in each module")
parser.add_argument('--filter_factor',default=2,type=float,help="set filters to this raised to the current module index")
parser.add_argument('--hidden_size',default=0,type=int,help="size of hidden layer, if zero then no hidden layer")
parser.add_argument('--pool_type',default='max',choices=('max','avg'),help='type of pool to use between modules')
parser.add_argument('--activation_function', default='relu', choices=('relu','elu','sigmoid','lrelu'),help='non-linear activation function to use')
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
args = parser.parse_args()

class View(nn.Module):
	def __init__(self,shape):
		super(View, self).__init__()
		self.shape = shape
	def forward(self, input):
		return input.view(*self.shape)

class Net(nn.Module):
	def __init__(self, dims):
		super(Net, self).__init__()
		self.modules = []
		self.residuals = []
		nchannels = dims[0]
		ksize = args.module_kernel_size
		pad = ksize//2
		fmult = 1
		div = 1
		self.dropout = nn.Dropout(p=args.dropout)

		#select activation function
		func = F.relu
		if args.activation_function == 'elu':
			func = F.elu
		elif args.activation_function == 'sigmoid':
			func = F.sigmoid
		elif args.activation_function == 'lrelu':
			func = F.leaky_relu
		
		inmultincr = 0
		if args.module_connect == 'dense':
			inmultincr=1

		for m in range(args.num_modules):
			module = []
			inmult = 1
			filters = int(args.module_filters*fmult)
			startchannels = nchannels
			for i in range(args.module_depth):
				conv = nn.Conv3d(nchannels*inmult, filters, kernel_size=ksize,padding=pad)
				inmult += inmultincr
				self.add_module('conv_{}_{}'.format(m,i),conv)
				module.append(conv)
				module.append(func)
				nchannels = filters
			if args.module_connect == 'residual':
				conv = nn.Conv3d(startchannels,nchannels, kernel_size=1,padding=0)
				self.add_module('resconv_{}'.format(m),conv)
				self.residuals.append(conv)
			if m < args.num_modules:
				pool = nn.MaxPool3d(2)
				if args.pool_type == 'avg':
					pool = nn.AvgPool3d(2)
				self.add_module('pool_{}'.format(m),pool)
				module.append(pool)
				div *= 2
			
			self.modules.append(module)
			fmult *= args.filter_factor
		last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * filters)
		lastmod = []
		lastmod.append(View((-1,last_size)))

		if args.hidden_size > 0:
			self.add_module("linear_dropout",self.dropout)
			lastmod.append(self.dropout)
			fc = nn.Linear(last_size, args.hidden_size)
			self.add_module('hidden_fc',fc)
			lastmod.append(fc)
			lastmod.append(func)
			last_size = args.hidden_size

		self.modules.append(lastmod)

		self.fc = nn.Linear(last_size,1)
		self.add_module('last_fc',self.fc)

	def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
		isdense = False
		isres = False
		if args.module_connect == 'dense':
			isdense = True
		if args.module_connect == 'residual':
			isres = True

		for (m,module) in enumerate(self.modules):
			preconvs = []
			if isres and len(self.residuals) > m:
				passthrough = self.residuals[m](x)
			else:
				isres = False
			for (l,layer) in enumerate(module):
				if isinstance(layer,nn.Conv3d) and isdense:
					if preconvs:
						x = torch.cat((x,*preconvs),1)
				if isres and l == len(module)-1:
					x = x + passthrough
				x = layer(x)

				if isinstance(layer,nn.Conv3d) and isdense:
					preconvs.append(x)

		return x

	def forward(self,x1,x2):
		lig1 = self.forward_one(x1)
		lig2 = self.forward_one(x2)
		diff = self.dropout(lig1 - lig2)
		return F.sigmoid(self.fc(diff))

def weights_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight.data)
		if m.bias is not None:
			init.constant_(m.bias.data,0)

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

def train(model, traine, optimizer, epoch, size):
	model.train()
	train_loss = 0
	correct = 0
	total = 0
	output_dist,actual = [], []

	for _ in range(size[0]):
		batch_1 = traine.next_batch(batch_size)
		gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
		batch_2 = traine.next_batch(batch_size)
		gmaker.forward(batch_2, input_tensor_2,random_translation=2.0, random_rotation=True) 
		batch_1.extract_label(0, float_labels)
		labels = torch.unsqueeze(float_labels,1).float().to('cuda')
		optimizer.zero_grad()
		output = model(input_tensor_1,input_tensor_2)
		loss = criterion(output,labels)
		pred_label = get_labels(output, threshold)
		correct += (pred_label == labels).sum().item()
		total += batch_size
		train_loss += loss
		loss.backward()
		optimizer.step()
		output_dist += output.flatten().tolist()
		actual += labels.flatten().tolist()

   # if size[1] != 0: #make sure go through every single training example
   #	 batch_1 = traine.next_batch()
   #	 gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
   #	 batch_2 = traine.next_batch()
   #	 gmaker.forward(batch_2, input_tensor_2,random_translation=2.0, random_rotation=True) 
   #	 batch_1.extract_label(1, float_labels)
   #	 loss = criterion(output,labels)
   #	 train_loss += loss
   #	 loss.backward()
   #	 optimizer.step()
   #	 output_dist += output.flatten().tolist()
   #	 actual += labels.flatten().tolist()

	return train_loss/total, (correct/total), output_dist

def test(model, teste, size):
	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	tp,fp,tn,fn = 0,0,0,0
	tot_auc = 0
	output_dist,actual = [],[]

	with torch.no_grad():
		for _ in range(size[0]):	
			batch_1 = traine.next_batch(batch_size)
			gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
			batch_2 = traine.next_batch(batch_size)
			gmaker.forward(batch_2, input_tensor_2,random_translation=2.0, random_rotation=True) 
			batch_1.extract_label(0, float_labels)
			labels = torch.unsqueeze(float_labels,1).float().to('cuda')
			optimizer.zero_grad()
			output = model(input_tensor_1,input_tensor_2)
			loss = criterion(output,labels)

			pred_label= get_labels(output, threshold)
			correct += (pred_label == labels).sum().item()
			tp,fp,tn,fn = confusion(pred_label,labels,tp,fp,tn,fn)
			tot_auc += get_AUC(output, labels)
			total += batch_size
			test_loss += loss
			output_dist += output.flatten().tolist()
			actual += labels.flatten().tolist()

	return test_loss/total, (tp+tn)/(tp+fp+tn+fn), (tp)/(tp+fp), (tp)/(tp+fn), tot_auc/(total/batch_size),output_dist

tgs = ['two_legged'] + args.tags
wandb.init(entity='andmcnutt', project='DDG_model',config=args, tags=tgs)

#Parameters that are not important for hyperparameter sweep
batch_size=32
epochs=args.epoch
threshold=0.5

print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))

traine = molgrid.ExampleProvider(ligmolcache=args.ligtr,recmolcache=args.rectr, balanced=True,shuffle=True, max_group_size=2, group_batch_size=batch_size)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte,shuffle=True, max_group_size=2, group_batch_size=batch_size)
teste.populate(args.testfile)

trsize = traine.size()
tssize = teste.size()
one_e_tr = int(trsize/batch_size)
leftover_tr = trsize % batch_size
one_e_tt = int(tssize/batch_size)
leftover_tt = tssize % batch_size

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(14*2) #only one rec+onelig per example
tensor_shape = (batch_size,)+dims
tr_nums=(one_e_tr,leftover_tr,trsize)
tt_nums=(one_e_tt,leftover_tt, tssize)

model = Net(dims)
if torch.cuda.device_count() > 1:
	print("Using {} GPUs".format(torch.cuda.device_count()))
	model = nn.DataParallel(model)
else:
	print('GPUS: {}'.format(torch.cuda.device_count()))
model.to('cuda:0')
model.apply(weights_init)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weightdecay)
if args.solver=="adam":
	optimizer=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.00001, factor=0.5, patience=5, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)

if leftover_tr != 0:
	tr_leftover_shape = (leftover_tr,) + dims
	tr_l_tensor = torch.zeros(tr_leftover_shape, dtype=torch.float32, device='cuda')
	tr_l_labels = torch.zeros(leftover_tr, dtype=torch.float32)
if leftover_tt != 0:
	tt_leftover_shape = (leftover_tt,) + dims
	tt_l_tensor = torch.zeros(tt_leftover_shape, dtype=torch.float32, device='cuda')
	tt_l_labels = torch.zeros(leftover_tt, dtype=torch.float32)

wandb.watch(model,log='all')

for epoch in range(epochs):
	tr_loss, tr_acc, out_dist =  train(model, traine, optimizer, epoch,tr_nums)
	tt_loss, tt_acc, tt_pr, tt_rec, auc, out_d = test(model, teste, tt_nums)
	scheduler.step(tr_loss)
	
	wandb.log({"Output Distribution Train": wandb.Histogram(np.array(out_dist))}, commit=False)
	wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d))}, commit=False)
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
print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
