import re
import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch import autograd
import wandb
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--ligtr', required=True, help='location of training ligand cache file input')
parser.add_argument('--rectr', required=True,help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=True, help='location of training information, this must have a group indicator')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information, this must have a group indicator')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--non_lin',choices=['relu','leakyrelu'],default='relu',help='non-linearity to use in the network')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="sgd", choices=('adam','sgd'), type=str, help="solver to use")
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--extra_stats',default=False,action='store_true',help='keep statistics about per receptor R values') 
parser.add_argument('--use_model','-m',default='paper',choices=['paper','default2018','extend_default2018'],help='Network architecture to use')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=128,type=int,help='size of fully connected layer before subtraction in latent space')
args = parser.parse_args()

if  args.use_model == 'paper':
    from paper_model import Net
elif args.use_model == 'default2018':
    from default2018_model import Net
elif args.use_model == 'extend_default2018':
    from extended_default2018_model import Net

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
                if args.clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(),args.clip)
                optimizer.step()
                output_dist += output.flatten().tolist()
                actual += labels.flatten().tolist()

        try:
            r=pearsonr(np.array(actual),np.array(output_dist))
        except ValueError as e:
            print('{}:{}'.format(epoch,e))
            r=[np.nan,np.nan]
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
        return train_loss/(size[2]), output_dist, r[0], rmse,actual

def test(model, test_data, size, test_recs_split):
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

        # Calculating "Average" Pearson's R across each receptor
        last_val,r_ave= 0,0
        r_per_rec = dict()
        for test_rec, test_count in test_recs_split.items():
            r_rec, _ = pearsonr(np.array(actual[last_val:last_val+test_count]),np.array(output_dist[last_val:last_val+test_count]))
            r_per_rec[test_rec]=r_rec
            r_ave += r_rec
            last_val += test_count
        r_ave /= len(test_recs_split)

        try:
            r=pearsonr(np.array(actual),np.array(output_dist))
        except ValueError as e:
            print('{}:{}'.format(epoch,e))
            r=[np.nan,np.nan]
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
        return test_loss/(size[2]), output_dist,r[0],rmse,actual,r_ave,r_per_rec

tgs = ['two_legged'] + args.tags
wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

#Parameters that are not important for hyperparameter sweep
batch_size=16
epochs=args.epoch

print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr,recmolcache=args.rectr,balanced=True,shuffle=True, duplicate_first=True)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte,shuffle=True, duplicate_first=True)
teste.populate(args.testfile)
# To compute the "average" pearson R per receptor, count the number of pairs for each rec then iterate over that number later during test time
test_exs_per_rec=dict()
with open(args.testfile) as test_types:
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

trsize = traine.size()
tssize = teste.size()
one_e_tr = int(trsize/batch_size)
leftover_tr = trsize % batch_size
one_e_tt = int(tssize/batch_size)
leftover_tt = tssize % batch_size

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*4) #only one rec+onelig per example
tensor_shape = (batch_size,)+dims
tr_nums=(one_e_tr,leftover_tr,trsize)
tt_nums=(one_e_tt,leftover_tt, tssize)

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims,args)
if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
else:
        print('GPUS: {}'.format(torch.cuda.device_count()), flush=True)
model.to('cuda:0')
model.apply(weights_init)

if args.use_weights is not None:  #using the weights from an external source, only some of the network layers need to be the same
    print('about to use weights')
    pretrained_state_dict = torch.load(args.use_weights)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
if args.freeze_arms:
    for name,param in model.named_parameters():
        if 'conv' in name:
            print(name)
            param.requires_grad = False
    

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.solver=="adam":
        optimizer=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, verbose=True)

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
print('extra stats:{}'.format(args.extra_stats))
print('training now')
## I want to see how the model is doing on the test before even training, mostly for the pretrained models
tt_loss, out_d, tt_r, tt_rmse,tt_act, tt_rave,tt_r_per_rec = test(model, teste, tt_nums, test_exs_per_rec)
print(f'Before Training at all:\n\tTest Loss: {tt_loss}\n\tTest R:{tt_r}\n\tTest RMSE:{tt_rmse}')
for epoch in range(1,epochs+1):
        tr_loss, out_dist, tr_r, tr_rmse,tr_act = train(model, traine, optimizer, epoch,tr_nums)
        tt_loss, out_d, tt_r, tt_rmse,tt_act, tt_rave,tt_r_per_rec = test(model, teste, tt_nums, test_exs_per_rec)
        scheduler.step(tr_loss)
        
        wandb.log({"Output Distribution Train": wandb.Histogram(np.array(out_dist))}, commit=False)
        wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d))}, commit=False)
        if epoch % 10 == 0: # only log the graphs every 10 epochs, make things a bit faster
            fig = plt.figure(1)
            fig.clf()
            plt.scatter(tr_act,out_dist)
            plt.xlabel('Actual DDG')
            plt.ylabel('Predicted DDG')
            wandb.log({"Actual vs. Predicted DDG (Train)": fig}, commit=False)
            test_fig = plt.figure(2)
            test_fig.clf()
            plt.scatter(tt_act,out_d)
            plt.xlabel('Actual DDG')
            plt.ylabel('Predicted DDG')
            wandb.log({"Actual vs. Predicted DDG (Test)": test_fig}, commit=False)
            if args.extra_stats:
                rperr_fig = plt.figure(3)
                rperr_fig.clf()
                sorted_test_rperrec = dict(sorted(tt_r_per_rec.items(), key=lambda item: item[0]))
                rec_pdbs, rvals = list(sorted_test_rperrec.keys()),list(sorted_test_rperrec.values())
                plt.bar(list(range(len(rvals))),rvals,tick_label=rec_pdbs)
                plt.ylabel("Pearson's R value")
                wandb.log({"R Value Per Receptor (Test)": rperr_fig},commit=False)
                rvsnligs_fig=plt.figure(4)
                rvsnligs_fig.clf()
                sorted_num_ligs = dict(sorted(test_exs_per_rec.items(),key=lambda item: item[0]))
                num_ligs = list(sorted_num_ligs.values())
                plt.scatter(num_ligs,rvals)
                plt.xlabel('number of ligands (test)')
                plt.ylabel("Pearson's R")
                wandb.log({"R Value Per Num_Ligs (Test)": rvsnligs_fig},commit=False)


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
                combined_data = np.array([out_d,tt_act]).T
                np.savetxt("out_actual_ddg.csv", combined_data, delimiter=",")
                wandb.save("out_actual_ddg.csv")
torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
