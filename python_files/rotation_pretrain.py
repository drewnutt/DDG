## Will attempt to use a model trained on all of BDB except for a PFAM
## Then pretrain that model on the rotation loss of the excluded PFAM
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
parser.add_argument('--solver', default="adam", choices=('adam','sgd'), type=str, help="solver to use")
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--rotation_epoch',default=30,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--extra_stats',default=False,action='store_true',help='keep statistics about per receptor R values') 
parser.add_argument('--use_model','-m',default='paper',choices=['paper', 'def2018', 'extend_def2018', 'multtask_def2018','ext_mult_def2018'], help='Network architecture to use')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=128,type=int,help='size of fully connected layer before subtraction in latent space')
parser.add_argument('--absolute_dg_loss', '-L',action='store_true',default=False,help='use a loss function (and model architecture) that utilizes the absolute binding affinity')
parser.add_argument('--rotation_loss_weight','-R',default=1.0,type=float,help='weight to use in adding the rotation loss to the other losses (default: %(default)d)')
parser.add_argument('--consistency_loss_weight','-C',default=1.0,type=float,help='weight to use in adding the consistency term to the other losses (default: %(default)d')
parser.add_argument('--absolute_loss_weight','-A',default=1.0,type=float,help='weight to use in adding the absolute loss terms to the other losses (default: %(default)d')
parser.add_argument('--ddg_loss_weight','-D',default=1.0,type=float,help='weight to use in adding the DDG loss terms to the other losses (default: %(default)d')
parser.add_argument('--run_model','-rm',required=True,help='run_path from wandb')
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
assert (args.absolute_dg_loss and args.use_model in ['multtask_def2018', 'ext_mult_def2018']) or (not args.absolute_dg_loss and args.use_model in ['paper','def2018','extend_def2018']), 'cannot have multitask loss with a non-multitask model'

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
    if isinstance(m, nn.conv3d) or isinstance(m, nn.linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not none:
            init.constant_(m.bias.data,0)

def train_rotation(model, traine, optimizer, epoch):
    model.train()
    full_loss = 0

    for idx, batch in enumerate(traine):
        if idx >= args.max_train_iter:
            break
        gmaker.forward(batch, input_tensor_1,random_translation=2.0, random_rotation=True) 
        gmaker.forward(batch, input_tensor_2,random_translation=2.0, random_rotation=True) 
        optimizer.zero_grad()
        ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:,:28,:,:,:],input_tensor_2[:,:28,:,:,:]) #Same rec-lig pair input to both arms, just rotated/translated differently
        ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:,28:,:,:,:],input_tensor_2[:,28:,:,:,:]) #Repeated for the second ligand
        rotation_loss = ddgrotloss1(ddg_lig1,torch.zeros(ddg_lig1.size(),device='cuda')) 
        rotation_loss += ddgrotloss2(ddg_lig2,torch.zeros(ddg_lig1.size(),device='cuda')) 
        rotation_loss += dgrotloss1(dg1_lig1,dg2_lig1) 
        rotation_loss += dgrotloss2(dg1_lig2,dg2_lig2) 
        loss = args.rotation_loss_weight * rotation_loss
        full_loss += loss
        loss.backward()
        optimizer.step()

    total_samples = (idx + 1) * len(batch) 
    avg_loss = full_loss/(total_samples)
    return avg_loss


def train_w_ss(model, train_data,test_data, optimizer, latent_rep):
    model.train()
    full_loss, lig_loss, rot_loss, DDG_loss, ss_loss = 0, 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    for idx, test_batch in enumerate(test_data):
        # Now do a batch of the self-supervised training on the unlabelled test data
        optimizer.zero_grad()
        gmaker.forward(test_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(test_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        if latent_rep:
            output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            self_super_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            self_super_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        self_super_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        self_super_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        ss_loss += self_super_loss 
        self_super_loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()

        # now do a batch of actual training with labels
        train_batch = train_data.next_batch()
        gmaker.forward(train_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(train_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        train_batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        train_batch.extract_label(2, lig1_label)
        train_batch.extract_label(3, lig2_label)
        lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
        lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
        if latent_rep:
            output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            rotation_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        loss_lig1 = criterion_lig1(lig1, lig1_labels)
        loss_lig2 = criterion_lig2(lig2, lig2_labels)
        ddg_loss = criterion(output, labels)
        rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2), output)
        lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
        lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
        lig_loss += loss_lig1 + loss_lig2
        rot_loss += rotation_loss
        DDG_loss += ddg_loss
        full_loss += loss
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()



        output_dist += output.flatten().tolist()
        actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(train_batch)
    try:
        r, _=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    try:
        rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = full_loss/(total_samples)
    avg_lig_loss = lig_loss / (2*total_samples)
    avg_rot_loss = rot_loss / (total_samples)
    avg_DDG_loss = DDG_loss / (total_samples)
    avg_ss_loss = ss_loss / total_samples
    tmp = avg_loss
    avg_loss = (tmp, avg_lig_loss, avg_DDG_loss, avg_rot_loss, avg_ss_loss)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels

def test(model, test_data, latent_rep,test_recs_split=None):
    model.eval()
    test_loss, lig_loss, rot_loss, DDG_loss = 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
            gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
            batch.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
            optimizer.zero_grad()
            batch.extract_label(2, lig1_label)
            batch.extract_label(3, lig2_label)
            lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
            lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
            if latent_rep:
                output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                rotation_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
                rotation_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
            else:
                output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
                rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
            loss_lig1 = criterion_lig1(lig1, lig1_labels)
            loss_lig2 = criterion_lig2(lig2, lig2_labels)
            ddg_loss = criterion(output, labels)
            rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
            rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
            loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2),output)
            lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
            lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
            lig_loss += loss_lig1 + loss_lig2
            rot_loss += rotation_loss
            DDG_loss += ddg_loss
            test_loss += loss
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(batch) 

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
    try:
        rligs, _=pearsonr(np.array(lig_pred), np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = float(test_loss)/(total_samples)
    avg_lig_loss = float(lig_loss) / (2*total_samples)
    avg_rot_loss = float(rot_loss) / (total_samples)
    avg_DDG_loss = float(DDG_loss) / (total_samples)
    tmp = avg_loss
    avg_loss = (tmp, avg_lig_loss, avg_DDG_loss, avg_rot_loss)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels, r_ave, r_per_rec


run.file('model.h5').download(replace=True)

#Parameters that are not important for hyperparameter sweep
batch_size=16
epochs=args.rotation_epoch

print(f'test_file={args.testfile}, lig_test={args.ligte}, rec_test={args.recte}')

if args.use_model == 'multtask_latent_def2018':
    latent_rep = True
else:
    latent_rep = False

traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, balanced=True, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
teste.populate(args.testfile)
ss_test = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
ss_test.populate(args.testfile)
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

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*4) #only one rec+onelig per example
tensor_shape = (batch_size,)+dims

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
    ddgrotloss1 = nn.MSELoss()
    ddgrotloss2 = nn.MSELoss()
    dgrotloss1  = nn.MSELoss()
    dgrotloss2  = nn.MSELoss()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)
if args.absolute_dg_loss:
    lig1_label = torch.zeros(batch_size, dtype=torch.float32)
    lig2_label = torch.zeros(batch_size, dtype=torch.float32)


wandb.watch(model,log='all')
print('extra stats:{}'.format(args.extra_stats))
tt_loss, out_d, tt_r, tt_rmse,tt_act, tt_rave,tt_r_per_rec = test(model, teste)
wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d))}, commit=False)
wandb.log({
    "Avg Test Loss Total": tt_loss[0],
    "Test R": tt_r[0],
    "Test RMSE": tt_rmse[0],
    "Avg Test Loss AbsAff": tt_loss[1],
    "Avg Test Loss DDG": tt_loss[2],
    "Avg Test Loss Rotation": tt_loss[3],
    "Test R AbsAff": float(tt_r[1]),
    "Test RMSE AbsAff": tt_rmse[1]})
print(f'Before Training at all:\n\tTest Loss: {tt_loss}\n\tTest R:{tt_r}\n\tTest RMSE:{tt_rmse}')
for epoch in range(1,epochs+1):
    tr_loss, out_dist, tr_r, tr_rmse, tr_act = train_w_ss(model, traine, ss_test, optimizer, latent_rep)
    tt_loss, out_d, tt_r, tt_rmse,tt_act, tt_rave,tt_r_per_rec = test(model, teste,latent_rep)

    wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d))}, commit=False)
    if epoch % 10 == 0: # only log the graphs every 10 epochs, make things a bit faster
        fig = plt.figure(1)
        fig.clf()
        plt.scatter(tr_act[0], out_dist[0])
        plt.xlabel('Actual DDG')
        plt.ylabel('Predicted DDG')
        wandb.log({"Actual vs. Predicted DDG (Train)": fig}, commit=False)
        test_fig = plt.figure(2)
        test_fig.clf()
        plt.scatter(tt_act[0], out_d[0])  # the first one is the ddg
        plt.xlabel('Actual DDG')
        plt.ylabel('Predicted DDG')
        wandb.log({"Actual vs. Predicted DDG (Test)": test_fig}, commit=False)
        train_absaff_fig = plt.figure(3)
        train_absaff_fig.clf()
        plt.scatter(tr_act[1],out_dist[1])
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Train)": train_absaff_fig})
        test_absaff_fig = plt.figure(4)
        test_absaff_fig.clf()
        plt.scatter(tt_act[1],out_d[1])
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Test)": test_absaff_fig})

    print(f'Test AbsAff R:{tt_r[1]:.4f}')
    wandb.log({
        "Avg Train Loss Total": tr_loss[0],
        "Avg Test Loss Total": tt_loss[0],
        "Train R": tr_r[0],
        "Test R": tt_r[0],
        #"Test 'Average' R": tt_rave,
        "Train RMSE": tr_rmse[0],
        "Test RMSE": tt_rmse[0],
        "Avg Train Loss AbsAff": tr_loss[1],
        "Avg Test Loss AbsAff": tt_loss[1],
        "Avg Train Loss DDG": tr_loss[2],
        "Avg Test Loss DDG": tt_loss[2],
        "Avg Train Loss Rotation": tr_loss[3],
        "Avg Self-Supervised Train Loss Rotation": tr_loss[4],
        "Avg Test Loss Rotation": tt_loss[3],
        "Train R AbsAff": float(tr_r[1]),
        "Test R AbsAff": float(tt_r[1]),
        "Train RMSE AbsAff": tr_rmse[1],
        "Test RMSE AbsAff": tt_rmse[1]})
    if not epoch % 50:
            torch.save(model.state_dict(), "model.h5")
            wandb.save('model.h5')
