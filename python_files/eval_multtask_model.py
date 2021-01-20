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
parser.add_argument('--use_model','-m',default='paper',choices=['paper', 'def2018', 'extend_def2018', 'multtask_def2018','ext_mult_def2018'], help='Network architecture to use')
parser.add_argument('--run_model','-rm',required=True,help='run_path from wandb')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=128,type=int,help='size of fully connected layer before subtraction in latent space')
parser.add_argument('--absolute_dg_loss', '-L',action='store_true',default=False,help='use a loss function (and model architecture) that utilizes the absolute binding affinity')
parser.add_argument('--consistency_term','-C',action='store_true',default=False,help='Use a consistency term in the multitask loss function to ensure absolute affinities agree with the relative binding affinities')
args = parser.parse_args()

tgs = ['two_legged'] + args.tags
wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

api = wandb.Api()
run = api.run(args.run_model)
ligte = args.ligte
recte = args.recte
testfile = args.testfile
args_dict = vars(args)
for k,v in run.config.items():
    args_dict[k] = v
args_dict['ligte'] = ligte
args_dict['recte'] = recte
args_dict['testfile'] = testfile
print(args)

wandb.config.update(args,allow_val_change=True)

print(args.absolute_dg_loss, args.use_model)
assert (args.absolute_dg_loss and args.use_model in ['multtask_def2018', 'ext_mult_def2018']) or (not args.absolute_dg_loss and args.use_model in ['paper','def2018','extend_def2018']), 'Cannot have multitask loss with a non-multitask model'

if  args.use_model == 'paper':
    from paper_model import Net
elif args.use_model == 'def2018':
    from default2018_model import Net
elif args.use_model == 'extend_def2018':
    from extended_default2018_model import Net
elif args.use_model == 'multtask_def2018':
    from multtask_def2018_model import Net
elif args.use_model == 'ext_mult_def2018':
    from extended_multtask_def2018_model import Net

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data,0)

# def train(model, traine, optimizer, epoch, size):
#     model.train()
#     train_loss = 0
#     lig_loss = 0

#     output_dist,actual = [], []
#     lig_pred,lig_labels = [] , []
#     for _ in range(size[0]):
#         batch_1 = traine.next_batch(batch_size)
#         gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
#         batch_1.extract_label(1, float_labels)
#         labels = torch.unsqueeze(float_labels,1).float().to('cuda')
#         optimizer.zero_grad()
#         if args.absolute_dg_loss:
#             batch_1.extract_label(2, lig1_label)
#             batch_1.extract_label(3, lig2_label)
#             lig1_labels = torch.unsqueeze(lig1_label,1).float().to('cuda')
#             lig2_labels = torch.unsqueeze(lig2_label,1).float().to('cuda')
#             output, lig1, lig2 = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
#             loss_lig1 = criterion_lig1(lig1,lig1_labels)
#             loss_lig2 = criterion_lig2(lig2,lig2_labels)
#             ddg_loss = criterion(output,labels)
#             loss = loss_lig1 + loss_lig2 + ddg_loss + int(args.consistency_term) * criterion((lig1-lig2),output)
#             lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
#             lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
#             lig_loss += loss_lig1 + loss_lig2
#         else:
#             output = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
#             loss = criterion(output,labels)
#         train_loss += loss
#         loss.backward()
#         if args.clip > 0:
#             nn.utils.clip_grad_norm_(model.parameters(),args.clip)
#         optimizer.step()
#         output_dist += output.flatten().tolist()
#         actual += labels.flatten().tolist()

#     try:
#         r, _=pearsonr(np.array(actual),np.array(output_dist))
#     except ValueError as e:
#         print('{}:{}'.format(epoch,e))
#         r=np.nan
#     if args.absolute_dg_loss:
#         try:
#             rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
#             temp = r
#             r = (temp,rligs)
#         except ValueError as e:
#             print(f'{epoch}:{e}')
#             tmp = r
#             r = (tmp,np.nan)
#     rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
#     avg_loss = train_loss/(size[2])
#     if args.absolute_dg_loss:
#         avg_lig_loss = lig_loss / (2*size[2])
#         tmp = avg_loss
#         avg_loss = (tmp,avg_lig_loss)
#         rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
#         tmp = rmse
#         rmse = (rmse, rmse_ligs)
#     return avg_loss, output_dist, r, rmse,actual

def test(model, test_data, size, test_recs_split=None):
    model.eval()
    test_loss = 0
    lig_loss = 0

    output_dist,actual = [],[]
    lig_pred,lig_labels = [] , []
    with torch.no_grad():
        for _ in range(size[0]):        
            batch_1 = test_data.next_batch(batch_size)
            gmaker.forward(batch_1, input_tensor_1,random_translation=2.0, random_rotation=True) 
            batch_1.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels,1).float().to('cuda')
            optimizer.zero_grad()
            if args.absolute_dg_loss:
                batch_1.extract_label(2, lig1_label)
                batch_1.extract_label(3, lig2_label)
                lig1_labels = torch.unsqueeze(lig1_label,1).float().to('cuda')
                lig2_labels = torch.unsqueeze(lig2_label,1).float().to('cuda')
                output,lig1,lig2 = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
                loss_lig1 = criterion_lig1(lig1,lig1_labels)
                loss_lig2 = criterion_lig2(lig2,lig2_labels)
                ddg_loss = criterion(output,labels)
                loss = loss_lig1 + loss_lig2 + ddg_loss + int(args.consistency_term) * criterion((lig1-lig2),output)
                lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
                lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
                lig_loss += loss_lig1 + loss_lig2
            else:
                output = model(input_tensor_1[:,:28,:,:,:],input_tensor_1[:,28:,:,:,:])
                loss = criterion(output,labels)
            test_loss += loss
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()

    # Calculating "Average" Pearson's R across each receptor
    if test_recs_split is not None:
        last_val,r_ave= 0,0
        r_per_rec = dict()
        for test_rec, test_count in test_recs_split.items():
            r_rec, _ = pearsonr(np.array(actual[last_val:last_val+test_count]),np.array(output_dist[last_val:last_val+test_count]))
            r_per_rec[test_rec]=r_rec
            r_ave += r_rec
            last_val += test_count
        r_ave /= len(test_recs_split)
    else:
        r_ave = 0
        r_per_rec = 0

    try:
        r,_=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    if args.absolute_dg_loss:
        try:
            rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
            temp = r
            r = (temp,rligs)
        except ValueError as e:
            print(f'{epoch}:{e}')
            tmp = r
            r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = test_loss/(size[2])
    if args.absolute_dg_loss:
        avg_lig_loss = lig_loss / (2*size[2])
        tmp = avg_loss
        avg_loss = (tmp,avg_lig_loss)
        rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
        tmp = rmse
        rmse = (tmp, rmse_ligs)
        tmp = output_dist
        output_dist = (tmp, lig_pred)
        tmp = actual
        actual = (tmp, lig_labels)
    return avg_loss, output_dist,r,rmse,actual,r_ave,r_per_rec


run.file('model.h5').download(replace=True)

#Parameters that are not important for hyperparameter sweep
batch_size=16
epochs=args.epoch

print(f'test_file={args.testfile}, lig_test={args.ligte}, rec_test={args.recte}')

teste = molgrid.ExampleProvider(ligmolcache=args.ligte,recmolcache=args.recte,shuffle=True, duplicate_first=True)
teste.populate(args.testfile)
# To compute the "average" pearson R per receptor, count the number of pairs for each rec then iterate over that number later during test time
# test_exs_per_rec=dict()
# with open(args.testfile) as test_types:
#     count = 0
#     rec = ''
#     for line in test_types:
#         line_args = line.split(' ')
#         newrec = re.findall(r'([A-Z0-9]{4})/',line_args[4])[0]
#         if newrec != rec:
#             if count > 0:
#                 test_exs_per_rec[rec] = count
#                 count = 1
#             rec = newrec
#         else:
#             count += 1

tssize = teste.size()
one_e_tt = int(tssize/batch_size)
leftover_tt = tssize % batch_size

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*4) #only one rec+onelig per example
tensor_shape = (batch_size,)+dims
tt_nums=(one_e_tt,leftover_tt, tssize)

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims,args)
model.load_state_dict(torch.load('model.h5'))
if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
else:
        print('GPUS: {}'.format(torch.cuda.device_count()), flush=True)
model.to('cuda:0')

# if args.use_weights is not None:  #using the weights from an external source, only some of the network layers need to be the same
#     print('about to use weights')
#     pretrained_state_dict = torch.load(args.use_weights)
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
# if args.freeze_arms:
#     for name,param in model.named_parameters():
#         if 'conv' in name:
#             print(name)
#             param.requires_grad = False
    

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.solver=="adam":
        optimizer=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()
if args.absolute_dg_loss: # Note these are the same loss functions as the main one, so in production these could jsut use the same, but this allows for different loss functions
    criterion_lig1 = nn.MSELoss()
    criterion_lig2 = nn.MSELoss()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)
if args.absolute_dg_loss:
    lig1_label = torch.zeros(batch_size, dtype=torch.float32)
    lig2_label = torch.zeros(batch_size, dtype=torch.float32)


if leftover_tt != 0:
        tt_leftover_shape = (leftover_tt,) + dims
        tt_l_tensor = torch.zeros(tt_leftover_shape, dtype=torch.float32, device='cuda')
        tt_l_labels = torch.zeros(leftover_tt, dtype=torch.float32)

wandb.watch(model,log='all')
print('extra stats:{}'.format(args.extra_stats))
tt_loss, out_d, tt_r, tt_rmse,tt_act, tt_rave,tt_r_per_rec = test(model, teste, tt_nums)
wandb.log({"Output Distribution DDG Test": wandb.Histogram(np.array(out_d[0]))}, commit=False)
wandb.log({"Output Distribution Affinity Test": wandb.Histogram(np.array(out_d[1]))}, commit=False)
plt.scatter(tt_act[0],out_d[0])
plt.xlabel('Actual DDG')
plt.ylabel('Predicted DDG')
wandb.log({"Actual vs. Predicted DDG": plt})
plt.figure(2)
plt.scatter(tt_act[1],out_d[1])
plt.xlabel('Actual affinity')
plt.ylabel('Predicted affinity')
wandb.log({"Actual vs. Predicted Affinity": plt})
wandb.log({
    "Avg Test Loss": tt_loss[0],
    "Test R": tt_r[0],
    #"Test 'Average' R": tt_rave,
    "Test RMSE": tt_rmse[0],
    "Avg Test Loss AbsAff": tt_loss[1],
    "Test R AbsAff": tt_r[1],
    "Test RMSE AbsAff": tt_rmse[1]})
print("Final Test DDG Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d[0]),np.var(out_d[0])))
print("Final Test Affinity Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d[1]),np.var(out_d[1])))
print(f'Avg Test Loss: {tt_loss[0]},Test R: {tt_r[0]},Test RMSE: {tt_rmse[0]},Avg Test Loss AbsAff: {tt_loss[1]},Test R AbsAff: {tt_r[1]},Test RMSE AbsAff: {tt_rmse[1]}')
