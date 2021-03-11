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
parser.add_argument('--trainfile', required=True, help='location of training information')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--non_lin',choices=['relu','leakyrelu'],default='relu',help='non-linearity to use in the network')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="adam", choices=('adam','sgd'), type=str, help="solver to use")
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--extra_stats',default=False,action='store_true',help='keep statistics about per receptor R values') 
parser.add_argument('--use_model','-m',default='def2018',choices=['def2018'], help='Network architecture to use')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=128,type=int,help='size of fully connected layer before subtraction in latent space')
parser.add_argument('--absolute_dg_loss', '-L',action='store_true',default=False,help='use a loss function (and model architecture) that utilizes the absolute binding affinity')
parser.add_argument('--self_supervised_test', '-S',action='store_true',default=False,help='Use the self supervised loss on the test files (no labels used)')
parser.add_argument('--rotation_loss_weight','-R',default=1.0,type=float,help='weight to use in adding the rotation loss to the other losses (default: %(default)d)')
parser.add_argument('--consistency_loss_weight','-C',default=1.0,type=float,help='weight to use in adding the consistency term to the other losses (default: %(default)d')
parser.add_argument('--absolute_loss_weight','-A',default=1.0,type=float,help='weight to use in adding the absolute loss terms to the other losses (default: %(default)d')
parser.add_argument('--ddg_loss_weight','-D',default=1.0,type=float,help='weight to use in adding the DDG loss terms to the other losses (default: %(default)d')
parser.add_argument('--train_type',default='no_SS', choices=['no_SS','SS_simult_before','SS_simult_after'],help='what type of training loop to use')
args = parser.parse_args()

print(args.absolute_dg_loss, args.use_model)

if args.use_model == 'def2018':
    from default2018_single_model import Net

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def train(model, traine, optimizer):
    model.train()
    full_loss = 0

    output_dist, actual = [], []
    for idx, batch in enumerate(traine):
        gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        batch.extract_label(0, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        output = model(input_tensor_1[:, :, :, :, :])
        ddg_loss = criterion(output, labels)
        loss = ddg_loss 
        full_loss += loss
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        output_dist += output.flatten().tolist()
        actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(batch)
    try:
        r, _=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = full_loss/(total_samples)
    both_calc_distr = (output_dist)
    both_labels = (actual)
    return avg_loss, both_calc_distr, r, rmse, both_labels


def test(model, test_data,test_recs_split=None):
    model.eval()
    test_loss= 0

    output_dist, actual = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
            batch.extract_label(0, float_labels)
            labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
            optimizer.zero_grad()
            output = model(input_tensor_1[:, :, :, :, :])
            ddg_loss = criterion(output, labels)
            loss = ddg_loss
            test_loss += loss
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(batch) 

    try:
        r,_=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = float(test_loss)/(total_samples)
    both_calc_distr = (output_dist)
    both_labels = (actual)
    return avg_loss, both_calc_distr, r, rmse, both_labels


# Make helper function to make meaningful tags
def make_tags(args):
    addnl_tags = []
    addnl_tags.append(args.use_model)
    if 'full_bdb' in args.ligtr:
        addnl_tags.append('full_BDB')
    addnl_tags.append(args.train_type)
    return addnl_tags


tgs = make_tags(args) + args.tags
wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

#Parameters that are not important for hyperparameter sweep
batch_size = 16
epochs = args.epoch

# print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, shuffle=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
teste.populate(args.testfile)
# To compute the "average" pearson R per receptor, count the number of pairs for each rec then iterate over that number later during test time
# test_exs_per_rec=dict()
# with open(args.testfile) as test_types:
#     count = 0
#     rec = ''
#     for lineuse a loss function (and model architecture) that utilizes the absolute binding affinity in test_types:
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
dims = gmaker.grid_dimensions(14*2)  # only one rec+onelig per example
tensor_shape = (batch_size,)+dims

actual_dims = (dims[0], *dims[1:])
model = Net(actual_dims,args)
if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
else:
        print('GPUS: {}'.format(torch.cuda.device_count()), flush=True)
model.to('cuda:0')
model.apply(weights_init)

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
    

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.solver == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)

wandb.watch(model, log='all')
print('extra stats:{}'.format(args.extra_stats))
print('training now')
## I want to see how the model is doing on the test before even training, mostly for the pretrained models
tt_loss, out_d, tt_r, tt_rmse, tt_act = test(model, teste)
print(f'Before Training at all:\n\tTest Loss: {tt_loss}\n\tTest R:{tt_r}\n\tTest RMSE:{tt_rmse}')
for epoch in range(1, epochs+1):
    # if args.self_supervised_test:
    #     ss_loss = train_rotation(model, teste, optimizer, latent_rep)
    # tr_loss, out_dist, tr_r, tr_rmse, tr_act = train(model, traine, optimizer, latent_rep)
    tr_loss, out_dist, tr_r, tr_rmse, tr_act = train(model, traine, optimizer)
    tt_loss, out_d, tt_r, tt_rmse, tt_act = test(model, teste)
    scheduler.step(tr_loss)
    
    if epoch % 10 == 0: # only log the graphs every 10 epochs, make things a bit faster
        train_absaff_fig = plt.figure(3)
        train_absaff_fig.clf()
        plt.scatter(tr_act,out_dist)
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Train)": train_absaff_fig})
        test_absaff_fig = plt.figure(4)
        test_absaff_fig.clf()
        plt.scatter(tt_act,out_d)
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Test)": test_absaff_fig})

    print(f'Test/Train AbsAff R:{tt_r:.4f}\t{tr_r:.4f}')
    wandb.log({
        #"Test 'Average' R": tt_rave,
        "Avg Train Loss AbsAff": tr_loss,
        "Avg Test Loss AbsAff": tt_loss,
        "Train R AbsAff": float(tr_r),
        "Test R AbsAff": float(tt_r),
        "Train RMSE AbsAff": tr_rmse,
        "Test RMSE AbsAff": tt_rmse})
    if not epoch % 50:
            torch.save(model.state_dict(), "model.h5")
            wandb.save('model.h5')
torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
# print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
# print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
