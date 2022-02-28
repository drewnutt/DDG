#!/usr/bin/env python3

import molgrid
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
from argparse import Namespace
from glob import glob

import torch


def get_stats(epoch, pred_ddg, actual_ddg, pred_abs, actual_abs):
    try:
        r_ddg, _ = pearsonr(np.array(actual_ddg), np.array(pred_ddg))
    except ValueError as e:
        print('{}:{}'.format(epoch, e))
        r_ddg = np.nan
    try:
        r_abs, _ = pearsonr(np.array(pred_abs), np.array(actual_abs))
    except ValueError as e:
        print(f'{epoch}:{e}')
        r_abs = np.nan
    r = (r_ddg, r_abs)

    try:
        sr, _ = spearmanr(np.array(actual_ddg), np.array(pred_ddg),axis=1)
    except ValueError as e:
        print('{}:{}'.format(epoch, e))
        sr = np.nan
    try:
        tau, _ = kendalltau(np.array(actual_ddg), np.array(pred_ddg))
    except ValueError as e:
        print(f'{epoch}:{e}')
        tau = np.nan

    rmse_ddg = np.sqrt(((np.array(pred_ddg)-np.array(actual_ddg)) ** 2).mean())
    rmse_abs = np.sqrt(((np.array(pred_abs)-np.array(actual_abs)) ** 2).mean())
    rmse = (rmse_ddg, rmse_abs)

    mae = (np.abs(np.array(pred_ddg)-np.array(actual_ddg)).mean(),
            np.abs(np.array(pred_abs)-np.array(actual_abs)).mean())
    return r, sr, tau, rmse, mae

def get_eval(args,model_file,return_vals=False,correlation=False):
    if args.use_model == "multtask_latent_def2018":
        from python_files.models.multtask_latent_def2018_model import Net
    elif args.use_model == "multtask_latent_dense":
        from python_files.models.multtask_latent_dense_model import Dense as Net
    elif args.use_model == "multtask_latent_def2018_concat":
        from python_files.models.multtask_latent_def2018_concat_model import Net
    elif args.use_model == "def2018":
        return get_eval_nosiam(args,model_file)

    batch_size=2
    test_data = molgrid.ExampleProvider(ligmolcache=f"cache/{args.ligte.split('/')[-1]}",
                                        recmolcache=f"cache/{args.recte.split('/')[-1]}", 
                                     duplicate_first=True, default_batch_size=batch_size,
                                    iteration_scheme=molgrid.IterationScheme.LargeEpoch)

    addnl_directory=''
    if 'pfam' in args.testfile:
        addnl_directory='PFAM_CV/'
    if 'jl' in args.testfile:
        addnl_directory='jimenez/'
    if 'external_test' in args.testfile:
        test_data.populate(args.testfile)
        print(args.testfile,test_data.size())
    else:
        test_data.populate(f"new_Prot_Lig_Valid/{addnl_directory}{args.testfile.split('/')[-1]}")
    
    gmaker = molgrid.GridMaker(binary=args.binary_rep)      
    dims = gmaker.grid_dimensions(14*4)  # only one rec+onelig per example      
    tensor_shape = (batch_size,)+dims      

    actual_dims = (dims[0]//2, *dims[1:])      
    model = Net(actual_dims,args).to('cuda')
    
    pretrained_state_dict = torch.load(model_file)      
    model_dict = model.state_dict()      
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}      
    model_dict.update(pretrained_dict)      
    model.load_state_dict(model_dict) 
    
    if "latent" in args.use_model:
        latent_rep = True
    else:
        latent_rep = False
        
    input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    float_labels = torch.zeros(batch_size, dtype=torch.float32)
    lig1_label = torch.zeros(batch_size, dtype=torch.float32)
    lig2_label = torch.zeros(batch_size, dtype=torch.float32)
    
    #running test loop
    model.eval()

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
            batch.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
            batch.extract_label(2, lig1_label)
            batch.extract_label(3, lig2_label)
            lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
            lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
            if latent_rep:
                output, lig1, lig2, lig1_rep1, lig2_rep1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
#                 if proj:
#                     lig1_rep1 = proj(lig1_rep1)
#                     lig1_rep2 = proj(lig1_rep2)
#                     lig2_rep1 = proj(lig2_rep1)
#                     lig2_rep2 = proj(lig2_rep2)
            else:
                output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
            lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()
            
    r, sr, tau, rmse, mae = get_stats(0,output_dist,actual,lig_pred,lig_labels)
    if return_vals == False:
        print('not return vals')
        return r,rmse,mae, sr, tau
    else:
        print('return vals')
        return r,rmse,mae,(output_dist,actual),sr,tau

def get_eval_nosiam(args,model_file):
#     if args.use_model == "def2018":
    from default2018_single_model import Net
    
    test_data = molgrid.ExampleProvider(ligmolcache=f"cache/{args.ligte.split('/')[-1]}",
                                        recmolcache=f"cache/{args.recte.split('/')[-1]}", 
                                    shuffle=True, duplicate_first=True, default_batch_size=16,
                                    iteration_scheme=molgrid.IterationScheme.LargeEpoch)      
    test_data.populate(f"new_Prot_Lig_Valid/{args.testfile.split('/')[-1]}")
    
    gmaker = molgrid.GridMaker(binary=args.binary_rep)      
    dims = gmaker.grid_dimensions(14*4)  # only one rec+onelig per example      
    tensor_shape = (16,)+dims      

    actual_dims = (dims[0]//2, *dims[1:])      
    model = Net(actual_dims,args).to('cuda')
    
    pretrained_state_dict = torch.load(model_file)      
    model_dict = model.state_dict()      
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}      
    model_dict.update(pretrained_dict)      
    model.load_state_dict(model_dict) 
        
    input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    float_labels = torch.zeros(16, dtype=torch.float32)
    lig1_label = torch.zeros(16, dtype=torch.float32)
    lig2_label = torch.zeros(16, dtype=torch.float32)
    
    #running test loop
    model.eval()

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
            batch.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
            batch.extract_label(2, lig1_label)
            batch.extract_label(3, lig2_label)
            lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
            lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
            lig1 = model(input_tensor_1[:, :28, :, :, :])
            lig2 = model(input_tensor_1[:, 28:, :, :, :])
            lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
            lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
            output_dist += (lig1-lig2).flatten().tolist()
            actual += labels.flatten().tolist()
            
    r, sr, tau, rmse, mae = get_stats(0,output_dist,actual,lig_pred,lig_labels)
    return r, rmse, mae

def create_stats(pub_api,model_save_path,num_runs = 25,model="multtask_latent_def2018",
                 num_addnl=1,weight_decay=0,ddg_weight=10,consistency_weight=1,
                absolute_weight=1,rotation_weight=1,use_weights=None,spec_name='',
                 rot_warmup=0,state="finished",tag='MseLoss',runs=None):
    if runs is None:
        runs = pub_api.runs(path='andmcnutt/DDG_model_Regression',
                           filters={"$and":[{"config.use_model":model},
                                            {'tags':f'addnl_ligs_{num_addnl}' if num_addnl else 'TrainAllPerms'},
                                            {"config.rot_warmup":rot_warmup},{"config.weight_decay":weight_decay},
                                            {"config.solver":"adam"},{"config.ddg_loss_weight":ddg_weight},
                                            {"config.consistency_loss_weight":consistency_weight},
                                            {"config.absolute_loss_weight":absolute_weight},
                                            {"config.rotation_loss_weight":rotation_weight},
                                            {"config.use_weights":use_weights},{"tags":tag},
                                            {"state":state}]})
    assert len(runs) == num_runs, f"only have {len(runs)} runs"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    rands = []
    stats_list = []
    for idx, run in enumerate(runs):
        if num_addnl:
            rand = run.config['trainfile'].split('_')[-3]
            assert "rand" in rand, f"{randval}, doesn't have 'rand' in it"
            randval = int(rand.replace('rand',''))
        else:
            randval = idx
        rands.append(randval)

        if not os.path.isfile(f"{model_save_path}model{spec_name}_{num_addnl}_{randval}.h5"):
            run.file('model.h5').download(root=model_save_path)
            os.rename(f"{model_save_path}model.h5",f"{model_save_path}model{spec_name}_{num_addnl}_{randval}.h5")

        config = Namespace()
        for k,v in run.config.items():
            setattr(config,k,v)
        
        if 'all_newdata.types' in config.trainfile:
            r,rmse, mae, sr, tau = get_eval(config,f"{model_save_path}model{spec_name}_{num_addnl}_{randval}.h5")
            stats_list.append([randval] + list(r[0].values()) + list(rmse[0].values())
                              + list(mae[0].values()) + list(r[1].values())
                              + list(rmse[1].values()) + list(mae[1].values()))
        else:
            r, rmse, mae, sr, tau = get_eval(config,f"{model_save_path}model{spec_name}_{num_addnl}_{randval}.h5")
            stats_list.append([randval,r[0],rmse[0],mae[0],r[1],rmse[1],mae[1]])
        print(spec_name,randval,r,rmse,mae)
        
    if 'all_newdata.types' in config.trainfile:
        final_stats = pd.DataFrame(stats_list, columns = ['randval'] + \
                [f"{meas}_{ext_set.split('/')[-2]}" for meas in ['PearsonR','RMSE','MAE','Abs_PearsonR','Abs_RMSE','Abs_MAE'] for ext_set in sorted(glob('external_test/*/')) if 'Input' not in ext_set])
    else:
        final_stats = pd.DataFrame(stats_list,columns=['randval','PearsonR','RMSE','MAE','Abs_PearsonR','Abs_RMSE','Abs_MAE'])
    if num_addnl:
        final_stats.to_csv(f"{model_save_path}statistics{spec_name}_{num_addnl}.csv",index=False,float_format="%.4f")
    else:
        final_stats.to_csv(f"{model_save_path}statistics{spec_name}.csv",index=False,float_format="%.4f")
    return final_stats

def build_graph(list_csvs,use='all',
            base_statistics = ['PearsonR','RMSE','MAE','Abs_PearsonR','Abs_RMSE','Abs_MAE']):

    avg_std_base_statistics = [f'Avg_{base}' for base in base_statistics] + [f'Std_{base}' for base in base_statistics]
    assert (type(use) is int) or use == 'all'
    stats_list = []
    for csv in list_csvs:
        data = pd.read_csv(csv)
        if use != 'all':
            data.sort_values('randval',inplace=True)
            data = data[:use]
        avgs = data[base_statistics].mean(axis=0)
        stds = data[base_statistics].std(axis=0)
        stats_list.append([csv] + avgs.values.tolist()+stds.values.tolist())
#     print(stats_list)
    return pd.DataFrame(stats_list,columns=['CSV_name']+avg_std_base_statistics)

def create_external_stats(model_save_path,num_runs = 25,model="multtask_latent_def2018",
                 num_addnl=1,weight_decay=0,ddg_weight=10,consistency_weight=1,
                absolute_weight=1,rotation_weight=1,use_weights=None,spec_name='',
                 rot_warmup=0,state="finished",tag='MseLoss',dset=None,spec_test=None,correlation=False, return_vals=False):
    runs = glob(f'{model_save_path}*_{spec_name}.h5')
    assert len(runs) == num_runs, f"only have {len(runs)} runs"
    rands = []
    stats_list = []
    for idx, run in enumerate(sorted(runs)):
        print(run)
        rand = run.split('/')[-1].split('_')[1]
        randval = int(rand.replace('rand',''))
        # else:
        #     rand = run.split('/')[-1].split('.')[0].split('_')[2]
        #     randval = int(rand.replace('rand',''))

        rands.append(randval)

        if dset:
            config = Namespace(testfile=f'external_test/{dset}/{dset}_DDG_TE.types',
                    ligte=f'cache/lig_{dset}.molcache2',
                    recte=f'cache/rec_{dset}.molcache2',
                    use_model=model,binary_rep=False,dropout=0,hidden_size=0)
            if spec_test:
                config.testfile = f"external_test/{config.testfile.split('/')[1]}/{config.testfile.split('/')[1]}_DDG_gt.types"
                print(config.testfile)
        else:
            config = Namespace(testfile=f'external_test/{ model_save_path.split("/")[-2] }/{ model_save_path.split("/")[-2] }_DDG_TE.types',
                    ligte=f'cache/lig_{ model_save_path.split("/")[-2] }.molcache2',
                    recte=f'cache/rec_{ model_save_path.split("/")[-2] }.molcache2',
                    use_model=model,binary_rep=False,dropout=0,hidden_size=0)
            if spec_test:
                config.testfile = f"external_test/{config.testfile.split('/')[1]}/{config.testfile.split('/')[1]}_DDG_{spec_test}TT.types"
                print(config.testfile)
            
        if return_vals == True:
            r, rmse, mae, vals, sr,tau = get_eval(config,run,correlation=correlation,return_vals=True)
        else:
            r, rmse, mae, sr,tau, = get_eval(config,run,correlation=correlation)
        curr_stats = [randval, r[0],rmse[0],mae[0],
                                r[1],rmse[1],mae[1]]
        if correlation == True:
            # [randval, r[0],rmse[0],mae[0],sr,tau,
            #                     r[1],rmse[1],mae[1]]
            curr_stats.insert(4,sr)
            curr_stats.insert(5,tau)
        if return_vals == True:
            curr_stats += [vals[0],vals[1]]
        stats_list.append(curr_stats)
    columns = ['randval','PearsonR','RMSE','MAE','Abs_PearsonR','Abs_RMSE','Abs_MAE']
    if correlation == True:
        # columns=['randval','PearsonR','RMSE','MAE','SpearmanR','Tau','Abs_PearsonR','Abs_RMSE','Abs_MAE'
        columns.insert(4,'SpearmanR')
        columns.insert(5,'Tau')
    if return_vals == True:
        columns += ['Predicted','Actual']

    final_stats = pd.DataFrame(stats_list,columns=columns)
    final_stats.to_csv(f"{model_save_path}statistics{spec_name}.csv",index=False,float_format="%.4f")
    return final_stats
