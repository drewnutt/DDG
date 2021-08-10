#!/usr/bin/env python

import numpy as np
import requests
import re
import itertools
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed','-s',default=42, type=int, help='seed for RNG (default=%(default)s)')
parser.add_argument('--train','-T',required=True,help='train file to add to')
parser.add_argument('--test','-E',required=True,help='test file to extract from')
parser.add_argument('--num_examples','-N',default=1, type=int, help='number of additional ligands to pull from the test group (0 is undefined), (default=%(default)s)')
# parser.add_argument('--output', '-o',required=True, help='basenames of train/test files')

args= parser.parse_args()
assert args.num_examples > 0

def get_permutes(dF):
    permuted_df = pd.DataFrame(columns=dF.columns)
    for idx, row in dF.iterrows():
        second_row = pd.Series([int(-row['reglabel']>1),-row['reglabel'],row['dg2'],row['dg1'],row['rec'],row['lig2'],row['lig1']],index=row.index) 
        permuted_df = permuted_df.append([row,second_row],ignore_index=True)
    return permuted_df.drop_duplicates(ignore_index=True)


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
possible_ligs.sort()
ref_lig = possible_ligs[0]
chosen_ligs = possible_ligs[rng.choice(np.arange(1,len(possible_ligs)),size=args.num_examples,replace=False)]
chosen_ligs = np.append(chosen_ligs,ref_lig)
extracted_combos = chosen_cong[(chosen_cong['lig1'].isin(chosen_ligs)) & (chosen_cong['lig2'].isin(chosen_ligs))].copy()
extracted_permutes = get_permutes(extracted_combos)
extracted_permutes['strat_label'] = 1
new_test = test_data[~((test_data['lig1'].isin(chosen_ligs)) & (test_data['lig2'].isin(chosen_ligs)))]
new_test = get_permutes(new_test)

combined_data = train_data.append(extracted_permutes, ignore_index=True)
combined_data.to_csv(f"train_{args.train.split('/')[-1].split('.')[0]}_{args.test.split('/')[-1].split('.')[0]}_{args.num_examples}.types", sep=' ', header=False, columns=['strat_label','reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2'],index=False, float_format='%0.4f')
new_test.to_csv(f"test_{args.train.split('/')[-1].split('.')[0]}_{args.test.split('/')[-1].split('.')[0]}_{args.num_examples}.types", sep=' ', header=False, columns=['clslabel','reglabel', 'dg1', 'dg2', 'rec', 'lig1', 'lig2'],index=False, float_format='%0.4f')
