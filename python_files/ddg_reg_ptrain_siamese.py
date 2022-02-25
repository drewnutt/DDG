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
parser.add_argument('--train_dataroot',default='.',help='location of additional training data')
parser.add_argument('--trainfile', required=True, help='location of training information, this must have a group indicator')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information, this must have a group indicator')
parser.add_argument('--stratify',default=False, action='store_true', help='use the column right before the receptor for stratification')
parser.add_argument('--stratify_rec','-S',default=False,action='store_true',help='toggle the training example provider stratifying by the receptor')
parser.add_argument('--no_rot_train',default=True,action='store_false',help='do not use random rotations during training')
parser.add_argument('--iter_scheme','-I',choices=['small','large'],default='small',help='what sort of epoch iteration scheme to use')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="adam", choices=('adam','sgd','lars','sam'), type=str, help="solver to use")
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--rho',default=0.05,type=float,help='rho to use with the Sharpness Aware Minimization (SAM) optimizer, size of the neighborhood')
parser.add_argument('--adaptive_SAM',default=False,action='store_true',help='use Adaptive Sharpness Aware Minimization')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')

parser.add_argument('--use_model','-m',default='paper',choices=['paper', 'latent_paper', 'def2018', 'extend_def2018', 'multtask_def2018','ext_mult_def2018', 'multtask_latent_def2018', 'multtask_latent_dense','multtask_latent_def2018_concat', 'multtask_latent_dense_concat','multtask_latent_equiv_def2018','multtask_latent_equiv2_def2018'], help='Network architecture to use')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=1024,type=int,help='size of fully connected layer before subtraction in latent space')
parser.add_argument('--batch_size',default=16,type=int,help='batch size (default: %(default)d)')

parser.add_argument('--absolute_dg_loss', '-L',action='store_true',default=False,help='use a loss function (and model architecture) that utilizes the absolute binding affinity')
parser.add_argument('--rotation_loss_weight','-R',default=1.0,type=float,help='weight to use in adding the rotation loss to the other losses (default: %(default)d)')
parser.add_argument('--consistency_loss_weight','-C',default=1.0,type=float,help='weight to use in adding the consistency term to the other losses (default: %(default)d')
parser.add_argument('--absolute_loss_weight','-A',default=1.0,type=float,help='weight to use in adding the absolute loss terms to the other losses (default: %(default)d')
parser.add_argument('--ddg_loss_weight','-D',default=1.0,type=float,help='weight to use in adding the DDG loss terms to the other losses (default: %(default)d')
parser.add_argument('--latent_loss',default='mse', choices=['mse','corr'],help='what type of loss to apply to the latent representations')
parser.add_argument('--rot_warmup','-RW',default=0,type=int,help='how many epochs to warmup from 0 to your desired weight for rotation loss')

parser.add_argument('--crosscorr_lambda', type=float, help='lambda value to use in the Cross Correlation Loss')
parser.add_argument('--proj_size',type=int,default=4096,help='size to project the latent representation to, this is the dimension that the CrossCorrLoss will be applied to (default: %(default)d')
parser.add_argument('--proj_layers',type=int,default=3,help='how many layers in the projection network, if 0 then there is no projection network(default: %(default)d')

parser.add_argument('--print_out','-P',default=False,action='store_true',help='print the labels and predictions during training')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--no_wandb',default=False, action='store_true', help='do not log run with wandb')
args = parser.parse_args()

if args.use_model == 'paper':
    from models.paper_model import Net
elif args.use_model == 'latent_paper':
    from models.paper_latent_model import Net
elif args.use_model == 'def2018':
    from models.default2018_model import Net
elif args.use_model == 'extend_def2018':
    from models.extended_default2018_model import Net
elif args.use_model == 'multtask_def2018':
    from models.multtask_def2018_model import Net
elif args.use_model == 'multtask_latent_def2018':
    from models.multtask_latent_def2018_model import Net
elif args.use_model == 'multtask_latent_def2018_concat':
    from models.multtask_latent_def2018_concat_model import Net
elif args.use_model == 'multtask_latent_dense_concat':
    from models.multtask_latent_dense_concat_model import Dense as Net
elif args.use_model == 'multtask_latent_dense':
    from models.multtask_latent_dense_model import Dense as Net
elif args.use_model == 'ext_mult_def2018':
    from models.extended_multtask_def2018_model import Net
elif args.use_model == 'multtask_latent_equiv_def2018':
    from models.multtask_latent_equiv_def2018_model import Net
elif args.use_model == 'multtask_latent_equiv2_def2018':
    from models.multtask_latent_equiv2_def2018_model import Net

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def train(model, traine, optimizer, latent_rep, epoch, proj=None):
    model.train()
    full_loss, lig_loss, rot_loss, DDG_loss = 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    for idx, batch in enumerate(traine):
        it = num_iters_pe * epoch + idx
        gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=args.no_rot_train) 
        gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        batch.extract_label(2, lig1_label)
        batch.extract_label(3, lig2_label)
        lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
        lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
        if latent_rep:
            output, lig1, lig2, lig1_rep1, lig2_rep1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            output2, lig1_2, lig2_2, lig1_rep2, lig2_rep2 = model(input_tensor_2[:, :28, :, :, :], input_tensor_2[:, 28:, :, :, :])
            # ddg_lig1, dg1_lig1, dg2_lig1, lig1_rep1, lig1_rep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            # ddg_lig2, dg1_lig2, dg2_lig2, lig2_rep1, lig2_rep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            if proj:
                lig1_rep1 = proj(lig1_rep1)
                lig1_rep2 = proj(lig1_rep2)
                lig2_rep1 = proj(lig2_rep1)
                lig2_rep2 = proj(lig2_rep2)
            rotation_loss = dgrotloss1(lig1_rep1, lig1_rep2)
            rotation_loss += dgrotloss2(lig2_rep1, lig2_rep2)
        else:
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
            rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda:0')) 
            rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda:0')) 
            del ddg_lig1, dg1_lig1, dg2_lig1, ddg_lig2, dg1_lig2, dg2_lig2 
            torch.cuda.empty_cache()
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
        loss_lig1 = criterion_lig1(lig1, lig1_labels)
        loss_lig2 = criterion_lig2(lig2, lig2_labels)
        ddg_loss = criterion(output, labels)
        loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight[it] * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2), output)
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
        if args.print_out:
            print(f'Train:{epoch}')
            for example in range(len(labels)):
                print(f"{labels[example].item():.3f} {output[example].item():.3f} {lig1_labels[example].item():.3f} {lig1[example].item():.3f} {lig2_labels[example].item():.3f} {lig2[example].item():.3f}")

    print(f'{epoch}: {args.rotation_loss_weight[it]}')

    total_samples = (idx + 1) * len(batch)
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
    tmp = avg_loss
    avg_loss = (tmp,avg_lig_loss,avg_DDG_loss,avg_rot_loss,None)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    mae = (np.abs(np.array(output_dist)-np.array(actual)).mean(), np.abs(np.array(lig_pred)-np.array(lig_labels)).mean())
    both_calc_distr = (output_dist,lig_pred)
    both_labels = (actual,lig_labels)
    return avg_loss, both_calc_distr, r, rmse, mae, both_labels

def test(model, test_data, latent_rep, epoch, proj=None):
    model.eval()
    test_loss, lig_loss, rot_loss, DDG_loss = 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            it = num_iters_pe * epoch + idx
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
                output, lig1, lig2, lig1_rep1, lig2_rep1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                output2, lig1_2, lig2_2, lig1_rep2, lig2_rep2 = model(input_tensor_2[:, :28, :, :, :], input_tensor_2[:, 28:, :, :, :])
                # ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                # ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                if proj:
                    lig1_rep1 = proj(lig1_rep1)
                    lig1_rep2 = proj(lig1_rep2)
                    lig2_rep1 = proj(lig2_rep1)
                    lig2_rep2 = proj(lig2_rep2)
                rotation_loss = dgrotloss1(lig1_rep1, lig1_rep2)
                rotation_loss += dgrotloss2(lig2_rep1, lig2_rep2)
            else:
                output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
                ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
                zero_tensor = torch.zeros(ddg_lig1.size()).to("cuda:0")
                rotation_loss += ddgrotloss1(ddg_lig1, zero_tensor) 
                rotation_loss += ddgrotloss2(ddg_lig2, zero_tensor) 
            loss_lig1 = criterion_lig1(lig1, lig1_labels)
            loss_lig2 = criterion_lig2(lig2, lig2_labels)
            ddg_loss = criterion(output, labels)
            loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight[it] * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2),output)
            lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
            lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
            lig_loss += loss_lig1 + loss_lig2
            rot_loss += rotation_loss
            DDG_loss += ddg_loss
            test_loss += loss
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()
            if args.print_out:
                print(f'Test:{epoch}')
                for example in range(len(labels)):
                    print(f"{labels[example].item():.3f} {output[example].item():.3f} {lig1_labels[example].item():.3f} {lig1[example].item():.3f} {lig2_labels[example].item():.3f} {lig2[example].item():.3f}")

    total_samples = (idx + 1) * len(batch) 

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
    mae = (np.abs(np.array(output_dist)-np.array(actual)).mean(), np.abs(np.array(lig_pred)-np.array(lig_labels)).mean())
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, mae, both_labels


# Make helper function to make meaningful tags
def make_tags(args):
    addnl_tags = []
    addnl_tags.append(args.use_model)
    if 'full_bdb' in args.ligtr:
        addnl_tags.append('full_BDB')
    addnl_tags.append(f'{args.latent_loss.title()}Loss')
    return addnl_tags


tgs = make_tags(args) + args.tags
if not args.no_wandb:
    wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

if "latent" in args.use_model:
    latent_rep = True
else:
    latent_rep = False
#Parameters that are not important for hyperparameter sweep
batch_size = args.batch_size
epochs = args.epoch

# print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))

iter_scheme = molgrid.IterationScheme.SmallEpoch
if args.iter_scheme == 'large':
    iter_scheme = molgrid.IterationScheme.LargeEpoch
if args.stratify:
    traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, data_root=args.train_dataroot, stratify_pos=0, stratify_min=0, stratify_max=1, stratify_step=.5, stratify_receptor=args.stratify_rec, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=iter_scheme)
else:
    traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, data_root=args.train_dataroot, stratify_receptor=args.stratify_rec, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=iter_scheme)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
teste.populate(args.testfile)

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*4)  # only one rec+onelig per example
tensor_shape = (batch_size,)+dims

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims,args)
if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
else:
        print('GPUS: {}'.format(torch.cuda.device_count()), flush=True)
model.to('cuda')
print('done moving model')
model.apply(weights_init)
print('applied weights')

if args.use_weights is not None:  # using the weights from an external source, only some of the network layers need to be the same
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
    
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion_lig1 = nn.MSELoss()
criterion_lig2 = nn.MSELoss()
# Not sure if these are replaceable with the Barlow Twins loss
ddgrotloss1 = nn.MSELoss()
ddgrotloss2 = nn.MSELoss()
projector = None
if args.latent_loss == 'mse':
    dgrotloss1 = nn.MSELoss()
    dgrotloss2 = nn.MSELoss()
elif latent_rep: ## only other option is 'covar' for the Barlow Twins approach, but requires latent rep
    _,_,_,rep1,rep2 = model(torch.rand(tensor_shape, device='cuda')[:, :28, :, :, :], torch.rand(tensor_shape,device='cuda')[:, 28:, :, :, :])
    init_size = rep1.shape[-1]
    assert init_size == rep2.shape[-1]
    proj_size = args.proj_size 
    if args.proj_layers:
        projector = Projector(init_size,proj_size,args.proj_layers)
        projector.to('cuda')
        projector.apply(weights_init)
        print(init_size,proj_size)
    else:
        proj_size = init_size
        print("not using any projector")
    crosscorr_lambda = 1.0/proj_size
    if args.crosscorr_lambda:
        crosscorr_lambda = args.crosscorr_lambda
    else:
        if not args.no_wandb:
           wandb.config.update({"crosscorr_lambda": crosscorr_lambda},allow_val_change=True) 
    dgrotloss1 = CrossCorrLoss(proj_size,crosscorr_lambda,device='cuda')
    dgrotloss2 = CrossCorrLoss(proj_size,crosscorr_lambda,device='cuda')
params = [param for param in model.parameters()]
if projector:
    print("Adding projector to optimizer parameters")
    params += [param for param in projector.parameters()]
closure = None
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
if args.solver == "sgd":
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.solver == 'lars':
    optimizer = LARS(params, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
elif args.solver == 'sam':
    raise NotImplementedError("Need to refactor code for SAM")
    assert (projector is None), "SAM does not work with more than one model parameters"
    from sam import SAM
    base_optim = optim.SGD
    optimizer = SAM(model.parameters(),base_optim, lr=args.lr,rho=args.rho,adaptive=args.adaptive_SAM)

num_iters_pe = int(np.ceil(traine.small_epoch_size()/args.batch_size))
if args.iter_scheme == 'large':
    num_iters_pe = int(np.floor(traine.large_epoch_size()/args.batch_size))
if args.rot_warmup:
    init_iters = num_iters_pe * (10)
    before_warmup = np.full((init_iters,),0)
    num_iters = num_iters_pe * args.rot_warmup
    warmup_schedule = np.linspace(0,args.rotation_loss_weight,int(num_iters) )
    final_iters = num_iters_pe * (args.epoch)
    args.rotation_loss_weight = np.concatenate((before_warmup,warmup_schedule,np.full((final_iters,),args.rotation_loss_weight)))
else:
    all_iters = num_iters_pe * (args.epoch) * 2
    args.rotation_loss_weight = np.full((all_iters,),args.rotation_loss_weight)



scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, threshold=0.001, patience=20, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)
lig1_label = torch.zeros(batch_size, dtype=torch.float32)
lig2_label = torch.zeros(batch_size, dtype=torch.float32)


if not args.no_wandb:
    wandb.watch(model, log='all')
# max_rot_weight = args.rotation_loss_weight
print('training now')
## I want to see how the model is doing on the test before even training, mostly for the pretrained models
tt_loss, out_d, tt_r, tt_rmse, tt_mae, tt_act = test(model, teste, latent_rep,1,proj=projector)
print(f'Before Training at all:\n\tTest Loss: {tt_loss}\n\tTest R:{tt_r}\n\tTest RMSE:{tt_rmse}\n\tTest MAE:{tt_mae}')
for epoch in range(1, epochs+1):
    tr_loss, out_dist, tr_r, tr_rmse, tr_mae, tr_act = train(model, traine, optimizer, latent_rep, epoch, proj=projector)
    tt_loss, out_d, tt_r, tt_rmse, tt_mae, tt_act = test(model, teste, latent_rep, epoch, proj=projector)

    scheduler.step(tr_loss[0])
    
    if not np.isnan(np.min(out_dist[0])) and not args.no_wandb:
        wandb.log({"Output Distribution Train": wandb.Histogram(np.array(out_dist[0]))}, commit=False)
    if not np.isnan(np.min(out_d[0])) and not args.no_wandb:
        wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d[0]))}, commit=False)
    if epoch % 10 == 0 and not args.no_wandb: # only log the graphs every 10 epochs, make things a bit faster
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
        wandb.log({"Actual vs. Predicted Affinity (Train)": train_absaff_fig}, commit=False)
        test_absaff_fig = plt.figure(4)
        test_absaff_fig.clf()
        plt.scatter(tt_act[1],out_d[1])
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Test)": test_absaff_fig}, commit=False)

    print(f'Test/Train AbsAff R:{tt_r[1]:.4f}\t{tr_r[1]:.4f}')
    if not args.no_wandb:
        wandb.log({
           "Avg Train Loss Total": tr_loss[0],
           "Avg Test Loss Total": tt_loss[0],
           "Train R": tr_r[0],
           "Test R": tt_r[0],
           "Train RMSE": tr_rmse[0],
           "Test RMSE": tt_rmse[0],
           "Train MAE": tr_mae[0],
           "Test MAE": tt_mae[0],
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
           "Test RMSE AbsAff": tt_rmse[1],
           "Train MAE AbsAff": tr_mae[1],
           "Test MAE AbsAff": tt_mae[1]})
    if not epoch % 50:
        torch.save(model.state_dict(), "model.h5")
        if not args.no_wandb:
            wandb.save('model.h5')
torch.save(model.state_dict(), "model.h5")
if not args.no_wandb:
    wandb.save('model.h5')
# print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
# print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
