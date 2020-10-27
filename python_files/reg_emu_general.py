import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import os,re
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
                nchannels = dims[0]

                self.func = F.relu
                self.dropout = nn.Dropout(p=args.dropout)

                avgpool1 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_0', avgpool1)
                self.modules.append(avgpool1)
                conv1 = nn.Conv3d(nchannels,out_channels=32,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit1_conv',conv1)
                self.modules.append(conv1)
                conv2 = nn.Conv3d(32,out_channels=32,padding=0,kernel_size=1,stride=1) 
                self.add_module('unit2_conv',conv2)
                self.modules.append(conv2)
                avgpool2 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_1', avgpool2)
                self.modules.append(avgpool2)
                conv3 = nn.Conv3d(32,out_channels=64,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit3_conv',conv3)
                self.modules.append(conv3)
                conv4 = nn.Conv3d(64,out_channels=64,padding=0,kernel_size=1,stride=1) 
                self.add_module('unit4_conv',conv4)
                self.modules.append(conv4)
                avgpool3 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_2', avgpool3)
                self.modules.append(avgpool3)
                conv5 = nn.Conv3d(64,out_channels=128,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit5_conv',conv5)
                self.modules.append(conv5)
                div = 2*2*2
                last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 128)
                print(last_size)
                flattener = View((-1,last_size))
                self.add_module('flatten',flattener)
                self.modules.append(flattener)
                self.fc = nn.Linear(last_size,1)
                self.add_module('last_fc',self.fc)

        def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
                for layer in self.modules:
                        x= layer(x)
                        if isinstance(layer,nn.Conv3d):
                                x=self.func(x)
                return x

        def forward(self,x1,x2):
                lig1 = self.forward_one(x1)
                lig2 = self.forward_one(x2)
                diff = self.dropout(lig1 - lig2)
                return self.fc(diff)

def weights_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                        init.constant_(m.bias.data,0)

def train(model, traine, optimizer, epoch, size):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        output_dist,actual = [], []
        for _ in range(size[0]):
                batch_1 = traine.next_batch(batch_size)
                gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
                batch_1.extract_label(1, float_labels)
                labels = torch.unsqueeze(float_labels,1).float().to('cuda')
                optimizer.zero_grad()
                output = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
                loss = criterion(output,labels)
                train_loss += loss
                loss.backward()
                optimizer.step()
                output_dist += output.flatten().tolist()
                actual += labels.flatten().tolist()

   # if size[1] != 0: #make sure go through every single training example
   #     batch_1 = traine.next_batch()
   #     gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
   #     batch_2 = traine.next_batch()
   #     gmaker.forward(batch_2, input_tensor_2,random_translation=2.0, random_rotation=True) 
   #     batch_1.extract_label(1, float_labels)
   #     loss = criterion(output,labels)
   #     train_loss += loss
   #     loss.backward()
   #     optimizer.step()
   #     output_dist += output.flatten().tolist()
   #     actual += labels.flatten().tolist()

        r=pearsonr(np.array(actual),np.array(output_dist))
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
        return train_loss/(size[2]), output_dist, r[0],rmse

def test(model, test_data, size,test_recs_split):
        model.eval()
        test_loss = 0

        output_dist,actual = [],[]
        with torch.no_grad():
                for _ in range(size[0]):        
                        batch_1 = test_data.next_batch(batch_size)
                        gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
                        batch_1.extract_label(1, float_labels)
                        labels = torch.unsqueeze(float_labels,1).float().to('cuda')
                        optimizer.zero_grad()
                        output = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
                        loss = criterion(output,labels)
                        test_loss += loss
                        output_dist += output.flatten().tolist()
                        actual += labels.flatten().tolist()
        last_val,r_ave= 0,0
        for test_count in test_recs_split:
            r_rec, _ = pearsonr(np.array(actual[last_val:last_val+test_count]),np.array(output_dist[last_val:last_val+test_count]))
            r_ave += r_rec
            last_val += test_count
        r_ave /= len(test_recs_split)
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
        r = pearsonr(np.array(actual),np.array(output_dist))
        return test_loss/(size[2]), output_dist,r[0],rmse,r_ave

tgs = ['two_legged'] + args.tags
wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

#Parameters that are not important for hyperparameter sweep
batch_size=32
epochs=args.epoch

print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr,recmolcache=args.rectr, balanced=True,shuffle=True, duplicate_first=True)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte, duplicate_first=True)
teste.populate(args.testfile)
test_exs_per_rec=list()
with open(args.testfile) as test_types:
	count=0
	rec=''
	for line in test_types:
		line_args = line.split(' ')
		newrec = re.findall(r'([A-Z0-9]{4})/',line_args[2])[0]
		if newrec != rec:
			rec = newrec
			if count > 0:
				test_exs_per_rec.append(count)
				count = 1
		else:
			count += 1


trsize = traine.size()
tssize = teste.size()
one_e_tr = int(trsize/batch_size)
leftover_tr = trsize % batch_size
one_e_tt = int(tssize/batch_size)
leftover_tt = tssize % batch_size

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(14*4) #only one rec+onelig per example
tensor_shape = (batch_size,)+dims
tr_nums=(one_e_tr,leftover_tr,trsize)
tt_nums=(one_e_tt,leftover_tt, tssize)

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims)
model.to('cuda:0')
model.apply(weights_init)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weightdecay)
if args.solver=="adam":
        optimizer=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
criterion = nn.MSELoss()
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
        tr_loss, out_dist, tr_r, tr_rmse = train(model, traine, optimizer, epoch,tr_nums)
        tt_loss, out_d, tt_r, tt_rmse, tt_rave  = test(model, teste, tt_nums,test_exs_per_rec)
        scheduler.step(tr_loss)
        
        wandb.log({"Output Distribution Train": wandb.Histogram(np.array(out_dist))}, commit=False)
        wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d))}, commit=False)
        wandb.log({
                "Avg Train Loss": tr_loss,
                "Avg Test Loss": tt_loss,
                "Train R": tr_r,
                "Test R": tt_r,
                "Test 'Average' R": tt_rave,
                "Train RMSE": tr_rmse, 
                "Test RMSE": tt_rmse})
        if not epoch % 50:
                torch.save(model.state_dict(), "model.h5")
                wandb.save('model.h5')
torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
