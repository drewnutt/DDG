#!/usr/bin/env python

import numpy as np
import requests
import re
import itertools
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed','-s',default=42, type=int, help='seed for RNG')
parser.add_argument('--train','-T',required=True,help='train file to add to')
parser.add_argument('--test','-E',required=True,help='test file to extract from')
parser.add_argument('--num_examples','-N',default=2, type=int, help='number of ligands to pull from the test group')
parser.add_argument('--output', '-o',required=True, help='basenames of train/test files')

args= parser.parse_args()

def make_csvs(train_data, groups, name, rec_to_pdb, pdb_to_ind):
    csv_out=pd.DataFrame()
    for val in groups:
        rec =rec_to_pdb[val]
        for r in rec:
            all_rec = train_data.loc[pdb_to_ind[r]]
            csv_out = csv_out.append(all_rec)
    csv_out.to_csv(name+'.txt',sep=' ',header=False,index=False)
    return csv_out.shape[0]

def outputTrainTest(train_test,strings,basename):
	for data,name in zip(train_test,strings):
		data.to_csv('{}_{}.types'.format(basename,name),sep=' ',header=False,columns=['label','reglabel','og', 'lig1','lig2'],index=False,float_format='%0.4f')


np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)

train_data = pd.read_csv(args.train, delimiter=' ', header=None)
train_data.columns = ['clslabel', 'reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2']
train_data['strat_label'] = 0

test_data = pd.read_csv(args.test, delimiter=' ', header=None)
test_data.columns = ['clslabel', 'reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2']
test_gp = test_data.groupby(by='rec')
sel_group = rng.integers(test_gp.ngroups)
chosen_cong = test_data[test_gp.ngroup() == sel_group]
possible_ligs = pd.unique(chosen_cong['lig1'])
chosen_ligs = possible_ligs[rng.integers(0,len(possible_ligs),size=args.num_examples)]
extracted_combos = chosen_cong[(chosen_cong['lig1'].isin(chosen_ligs)) & (chosen_cong['lig2'].isin(chosen_ligs))] 
extracted_combos['strat_label'] = 1
new_test = test_data[!test_data.isin(extracted_combos)]

combined_data = train_data.append(extracted_combos,ignore_index=True)
combined_data.to_csv(f"train_{args.train.split('/')[-1].split('.')[0]}_{args.test.split('/')[-1].split('.')[0]}_{args.num_examples}.types", sep=' ', header=False, columns=['strat_label','reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2'],index=False, float_format='%0.4f')
new_test.to_csv(f"test_{args.train.split('/')[-1].split('.')[0]}_{args.test.split('/')[-1].split('.')[0]}_{args.num_examples}.types", sep=' ', header=False, columns=['strat_label','reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2'],index=False, float_format='%0.4f')
