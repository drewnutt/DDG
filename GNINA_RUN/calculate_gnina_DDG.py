import argparse
import re
import csv
import pandas as pd
from itertools import permutations
import os
import sys
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

parser= argparse.ArgumentParser()
parser.add_argument('-gf', help='gnina run file')
parser.add_argument('--trainf',nargs='+',default=[], help='training input file for DDG model')
parser.add_argument('--model',required=True, help='gnina model used to calculate info')
parser.add_argument('--pre_compiled',action='store_true',default=False,help='all of the gnina information is compiled, just compare to Ground Truth Files')
args = parser.parse_args()

print((not args.pre_compiled and args.gf is not None))
print(args.pre_compiled and os.path.isfile(f'/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/gnina_DDG_{args.model}.txt'))
assert (not args.pre_compiled and args.gf is not None) or (args.pre_compiled and os.path.isfile(f'/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/gnina_DDG_{args.model}.txt')), 'Need gnina run file if you dont have precompiled DDG file for the given model'

def makeGninaDDGFile():
    ## first compile all of the ligand affinities computed by gnina
    ligand_list = []
    name_p = re.compile('(?:/)([A-Z0-9]+)(?:\.mol2)')
    logfiles=[]
    with open(args.gf, 'r') as f:
        for line in f:
            logfiles.append(line.split(' ')[-1].strip())

    base_dir='/net/dali/home/mscbio/anm329/deltadeltaG/'
    for logfile in logfiles:
        with open(base_dir+logfile) as f:
            for line in f:
                if 'Affinity' in line:
                    aff = float(re.findall(r'[\-\d.]+',line)[0])
                elif 'CNNscore' in line:
                    score = float(re.findall(r'[\d.]+',line)[0])
                elif 'CNNaffinity' in line:
                    caff = float(re.findall(r'[\d.]+',line)[0])
        name = logfile.replace(f'{args.model}','&').split('&')[1].split('.')[0].strip('_')
        ligand_list.append([name,caff,score,aff])
    gnina_out_df = pd.DataFrame(ligand_list,columns=['name','cnn_aff','cnn_score','vina_aff'])

    gnina_out_df['rec'] = gnina_out_df['name'].apply(lambda x: re.findall('[A-Z\d]{4}',x)[0])
    gnina_out_df.to_csv('test_affinities.csv')
    gnina_out_by_rec = gnina_out_df.groupby('rec')
    list_of_data = []
    for rec,group in gnina_out_by_rec:
        lig_perms = permutations(list(group.index),2)
        for lig1,lig2 in lig_perms:
            try:
                ## CNN outputs pK which should correlate with pIC50
                diffgaff = gnina_out_df.at[lig1,'cnn_aff'] - gnina_out_df.at[lig2,'cnn_aff']
                diffcnn_score = gnina_out_df.at[lig1,'cnn_score'] - gnina_out_df.at[lig2,'cnn_score']
                # Vina score is dG => we need to do -log10{exp(VINA/[R*T])} to get pK
                # Vina is in kcal/mol so we have R=1.987E-3 kcal/(K*mol) and T=293 => 0.582 kcal/mol
                diffPvaff = -np.log10(np.exp(gnina_out_df.at[lig1,'vina_aff']/0.582)) + np.log10(np.exp(gnina_out_df.at[lig2,'vina_aff']/0.582))
                info = [diffgaff,diffcnn_score,diffPvaff,rec,gnina_out_df.at[lig1,'name'],gnina_out_df.at[lig2,'name']]
                list_of_data.append(info)
            except Exception as e:
                print(rec,lig1,lig2, e)
    output_gnina_ddg = pd.DataFrame(list_of_data)
    output_gnina_ddg.columns = ['DDG_CNNaff', 'DDG_scr','DDG_Paff','rec', 'lig1', 'lig2']

    gnina_ddg = '/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/gnina_DDG_{}.txt'.format(args.model)
    output_gnina_ddg.to_csv(gnina_ddg, sep=' ', index=False)

    return output_gnina_ddg

def getDDGstats(compare_ddg_df,ground_truth_files):
    ligname_pattern = re.compile(r'(?:\/)([A-Z\d]{4}[\d]{2})(?:_.\.gninatypes)')
    comp_dict = dict()
    for trainfile in ground_truth_files:
        file_data = pd.read_csv(trainfile, sep=' ', header=None)
        train_data = file_data[[1,file_data.columns[-2],file_data.columns[-1]]].copy()
        train_data.columns = ['DDG','lig1','lig2']
        train_data['lig2'] = train_data['lig2'].apply(lambda x: re.findall(ligname_pattern,x)[0])
        train_data['lig1'] = train_data['lig1'].apply(lambda x: re.findall(ligname_pattern,x)[0])
        train_data['lig1_lig2'] = train_data['lig1'] + train_data['lig2']
        sorted_train = train_data.sort_values(by=['lig1_lig2'])
        subset_gnina_ddg = compare_ddg_df[compare_ddg_df['lig1_lig2'].isin(train_data['lig1_lig2'])]

        assert subset_gnina_ddg.shape[0] == train_data.shape[0]
        for col in subset_gnina_ddg.columns:
            if 'DDG' not in col:
                continue
            r,_ = pearsonr(subset_gnina_ddg[col].to_numpy(),sorted_train['DDG'].to_numpy())
            rmse = np.sqrt(((subset_gnina_ddg[col].to_numpy()-sorted_train['DDG'].to_numpy()) ** 2).mean())
            comp_dict[f'{trainfile.split("/")[-1]}_{col}_r'] = r
            comp_dict[f'{trainfile.split("/")[-1]}_{col}_rmse'] = rmse

    output_string = f'# {args.model}\n -----'
    for tfile in args.trainf:
        output_string += f'\n## {tfile}:\n'
        output_string += 'Metric | R | RMSE\n-----|-----|-----\nGNINA Affinity | {:.4f} | {:.2f} \nGNINA Score | {:.4f} | {:.2f}'.format(comp_dict[f'{tfile.split("/")[-1]}_DDG_CNNaff_r'], comp_dict[f'{tfile.split("/")[-1]}_DDG_CNNaff_rmse'], comp_dict[f'{tfile.split("/")[-1]}_DDG_scr_r'], comp_dict[f'{tfile.split("/")[-1]}_DDG_scr_rmse'])
        output_string += '\nVina Affinity | {:.4f} | {:.4f}'.format(comp_dict[f'{tfile.split("/")[-1]}_DDG_Paff_r'], comp_dict[f'{tfile.split("/")[-1]}_DDG_Paff_rmse'])
    model_stat = f'/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/{args.model}_stats.md'
    with open(model_stat,'w') as f:
        f.write(output_string)

if not args.pre_compiled:
   print(f"Creating Gnina DDG file from gf argument({args.gf}) and the logfiles output by the gnina runs")
   gnina_ddg_df = makeGninaDDGFile() 
else:
    print(f"Reading in Gnina DDG file (/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/gnina_DDG_{args.model}.txt)")
    gnina_ddg_df = pd.read_csv(f'/net/dali/home/mscbio/anm329/deltadeltaG/GNINA_RUN/gnina_DDG_{args.model}.txt', sep=' ')
gnina_ddg_df['lig1_lig2'] = gnina_ddg_df['lig1'].apply(lambda x:re.findall('[A-Z\d]{4}\d{2}',x)[0]) + gnina_ddg_df['lig2'].apply(lambda x:re.findall('[A-Z\d]{4}\d{2}',x)[0])
gnina_ddg_df = gnina_ddg_df.sort_values(by=['lig1_lig2'])
if len(args.trainf) > 0:
    print(f"Creating DDG stats for Gnina\nUsing these files for ground truth:\n\t{','.join(args.trainf)}\n")
    getDDGstats(gnina_ddg_df, args.trainf)
