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
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--ligtr', required=False, help='location of training ligand cache file input')
parser.add_argument('--rectr', required=False, help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=False, help='location of training information')
parser.add_argument('--ligte', required=False, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=False, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=False, help='location of testing information')
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
parser.add_argument('--run_model','-rm',required=True,help='run_path from wandb')
parser.add_argument('--eval_data_lig','-el',default=None,help='lig cache data to evaluate the trained model on')
parser.add_argument('--eval_data_rec','-er',default=None,help='rec cache data to evaluate the trained model on')
parser.add_argument('--eval_data_file','-ef',required=True,help='location of evaluation information, types file')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--extra_stats',default=False,action='store_true',help='keep statistics about per receptor R values') 
parser.add_argument('--wandbtags','-t',nargs='*',help='Tags for the wandb model')
#parser.add_argument('--regression',action='store_true',help='use a regression model instead of a classification model') need to add more to make this actually work for both classification and regression
args = parser.parse_args()
args.wandbtags.append('two_legged')
args.wandbtags.append('regression')
if len(args.wandbtags):
        wandb.init(entity='andmcnutt', project='DDG_model_Evaluation',name=os.path.basename(args.eval_data_file),tags=args.wandbtags)
else:
        wandb.init(entity='andmcnutt', project='DDG_model_Evaluation',name=os.path.basename(args.eval_data_file))
api = wandb.Api()
run = api.run(args.run_model)
args_dict = vars(args)
for k,v in run.config.items():
    args_dict[k] = v
print(args)
wandb.config.update(args)

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
                ksize = 3

                #select activation function
                self.func = F.relu
                if args.non_lin != 'relu':
                    self.func = F.leaky_relu
                if args.batch_norm:
                    self.bn1=nn.BatchNorm3d(32)
                    self.bn2=nn.BatchNorm3d(32)
                    self.bn3=nn.BatchNorm3d(3)
                else:
                    self.bn1 = None
                

                self.conv1 = nn.Conv3d(nchannels, 32, kernel_size=ksize)
                self.conv2 = nn.Conv3d(32,32,kernel_size=ksize)
                self.max1 = nn.MaxPool3d(3)
                
                self.conv3 = nn.Conv3d(32,3,kernel_size=ksize)
                self.lin1 = nn.Linear(5184,1,bias=False)
                self.dp1 = nn.Dropout(p=args.dropout)


        def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
                if self.bn1 is not None:
                    x= self.bn1(self.conv1(x))
                    x= self.bn2(self.conv2(self.func(x)))
                    x=self.max1(self.func(x))

                    x= self.func(self.bn3(self.conv3(x)))
                else:
                    x= self.conv1(x)
                    x= self.conv2(self.func(x))
                    x=self.max1(self.func(x))

                    x= self.func(self.conv3(x))

                x= x.view(x.shape[0], -1)       
                return x

        def forward(self,x1,x2):
                lig1 = self.forward_one(x1)
                lig2 = self.forward_one(x2)
                diff = self.dp1(lig1 - lig2)
                return self.lin1(diff)

def weights_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                        init.constant_(m.bias.data,0)

#def train(model, traine, optimizer, epoch, size):
#        model.train()
#        train_loss = 0
#        correct = 0
#        total = 0
#
#        output_dist,actual = [], []
#        for _ in range(size[0]):
#                batch_1 = traine.next_batch(batch_size)
#                gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
#                batch_1.extract_label(1, float_labels)
#                labels = torch.unsqueeze(float_labels,1).float().to('cuda')
#                optimizer.zero_grad()
#                output = model(input_tensor_1[:,:52,:,:,:],input_tensor_1[:,52:,:,:,:])
#                loss = criterion(output,labels)
#                train_loss += loss
#                loss.backward()
#                optimizer.step()
#                output_dist += output.flatten().tolist()
#                actual += labels.flatten().tolist()
#
#
#   # if size[1] != 0: #make sure go through every single training example
#   #     batch_1 = traine.next_batch()
#   #     gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
#   #     batch_2 = traine.next_batch()
#   #     gmaker.forward(batch_2, input_tensor_2,random_translation=2.0, random_rotation=True) 
#   #     batch_1.extract_label(1, float_labels)
#   #     loss = criterion(output,labels)
#   #     train_loss += loss
#   #     loss.backward()
#   #     optimizer.step()
#   #     output_dist += output.flatten().tolist()
#   #     actual += labels.flatten().tolist()
#
#        r=pearsonr(np.array(actual),np.array(output_dist))
#        return train_loss/(size[2]), output_dist, r[0]

def test(model, teste, size,test_recs_split):
        model.eval()
        test_loss = 0

        output_dist,actual = [],[]
        with torch.no_grad():
                for _ in range(size[0]):        
                        batch_1 = teste.next_batch(batch_size)
                        gmaker.forward(batch_1, input_tensor,random_translation=2.0, random_rotation=True) 
                        batch_1.extract_label(1, float_labels)
                        labels = torch.unsqueeze(float_labels,1).float().to('cuda')
                        output = model(input_tensor[:,:52,:,:,:],input_tensor[:,52:,:,:,:])
                        loss = criterion(output,labels)
                        test_loss += loss
                        output_dist += output.flatten().tolist()
                        actual += labels.flatten().tolist()

        # Calculating "Average" Pearson's R across each receptor
        last_val,r_ave= 0,0
        r_per_rec = dict()
        for test_rec, test_count in test_recs_split.items():
            r_rec, _ = pearsonr(np.array(actual[last_val:last_val+test_count]),np.array(output_dist[last_val:last_val+test_count]))
            r_per_rec[test_rec]=r_rec
            r_ave += r_rec
            last_val += test_count
        r_ave /= len(test_recs_split)

        r = pearsonr(np.array(actual),np.array(output_dist))
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
        return test_loss/(size[2]), output_dist,r[0], actual, rmse,r_ave



run.file('model.h5').download(replace=True)
#Parameters that are not important for hyperparameter sweep
batch_size=32
threshold=0.5

print('ligtr={}, rectr={}'.format(args.eval_data_lig,args.eval_data_rec))

teste = molgrid.ExampleProvider(molgrid.GninaVectorTyper(),balanced=True,shuffle=False, duplicate_first=True, data_root='separated_sets/')
if args.eval_data_lig is not None and args.eval_data_rec is not None:
    teste = molgrid.ExampleProvider(ligmolcache=args.eval_data_lig,recmolcache=args.eval_data_rec, balanced=True,shuffle=False, duplicate_first=True)
teste.populate(args.eval_data_file)
test_exs_per_rec=dict()
with open(args.eval_data_file) as test_types:
	count=0
	rec=''
	for line in test_types:
		line_args = line.split(' ')
		newrec = re.findall(r'([A-Z0-9]{4})/',line_args[2])[0]
		if newrec != rec:
			if count > 0:
				test_exs_per_rec[rec] = count
				count = 1
			rec = newrec
		else:
			count += 1

tssize = teste.size()
one_e_tt = int(tssize/batch_size)
leftover_tt = tssize % batch_size

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(molgrid.GninaVectorTyper().num_types()*4) #only one rec+one lig provided to each leg of network
tensor_shape = (batch_size,)+dims
tt_nums=(one_e_tt,leftover_tt, tssize)

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims)
model.load_state_dict(torch.load('model.h5'))
if torch.cuda.device_count() > 1:
    print("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
else:
    print('GPUS: {}'.format(torch.cuda.device_count()))
model.to('cuda:0')


input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)

if leftover_tt != 0:
    tt_leftover_shape = (leftover_tt,) + dims
    tt_l_tensor = torch.zeros(tt_leftover_shape, dtype=torch.float32, device='cuda')
    tt_l_labels = torch.zeros(leftover_tt, dtype=torch.float32)

wandb.watch(model,log='all')
criterion = nn.MSELoss()

tr_loss, out_dist, r, actual_dist, rmse, avg_r= test(model, teste, tt_nums, test_exs_per_rec)

wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_dist))}, commit=False)
plt.scatter(actual_dist,out_dist)
plt.xlabel('Actual DDG')
plt.ylabel('Predicted DDG')
wandb.log({"Actual vs. Predicted DDG": plt})
wandb.log({
    "Avg Test Loss": tr_loss,
    "Test R": r,
    "Test 'Average' R": avg_r,
    "Test RMSE": rmse})
