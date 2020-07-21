#!/usr/bin/env python

import numpy as np
import requests
import re
import itertools
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--random',action='store_true', help='Do a random split')
parser.add_argument('--seed','-s',default=42, type=int, help='seed for RNG')
parser.add_argument('--input','-i',required=True,help='full types file to decompose into train/test')
parser.add_argument('--output', '-o',help='basenames of train/test files,defaults to input name')

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
#url="https://www.bindingdb.org/validation_sets/index-1.jsp"
#r = requests.get(url)
#soup = BeautifulSoup(r.content,'html.parser')
#properstruct = re.compile('http://www.rcsb.org/pdb/explore/explore.do')
#
#rows = soup.find_all('tr')
#imp_rows = [i for i in rows if len(i.find_all('a')) > 0]
#
#rec_to_pdb = dict()
#list_of_pdbs=[]
#
#system = ""
#list_rec = []
#for idx, i in enumerate(imp_rows):
#    name = i.span
#    if name is not None:
#        if len(list_rec):
#            rec_to_pdb[system]=list_rec
#            list_rec = []
#        system = name.text
#        continue
#    #non-system names should make it here
#    if 'bold' in  i.attrs['class']:
#        pdbid = [j.text for j in i.find_all('a',href=properstruct)]
#        list_rec+= pdbid
#        list_of_pdbs+= pdbid
#if len(list_rec):
#    rec_to_pdb[system]=list_rec
#
#groups = [*rec_to_pdb]
#train = round(0.8*len(groups))
#val=round(1.0/3*len(groups))
#np.random.shuffle(groups)

train_data = pd.read_csv(args.input, delimiter=' ', header=None)
train_data.columns = ['label','reglabel','og', 'lig1','lig2']
train_data['recs'] = train_data['og'].astype(str).str[:4]
#train_data['rec'] = pd.Series(train_data.apply(lambda x: re.match(x,rec_of_row)[0],axis=1), index=train_data.index)
#pdb_to_ind = dict()
#for pdb in list_of_pdbs:
#    pdb_to_ind[pdb]=train_data.index[train_data['recs'] == pdb].tolist()
#del train_data['recs']
#
#perm1_train=groups[:train]
#remainder=groups[train:]
#perm1_val = remainder[:len(remainder)//2]
#perm1_test= remainder[len(remainder)//2:]
#print(len(perm1_train),len(perm1_val),len(perm1_test))
if args.random:
	shuffle_data = train_data.sample(frac=1).reset_index(drop=True)
	full_data_len = len(train_data)
	train = round(0.8*full_data_len)
	test = full_data_len - train
	train_data = shuffle_data[:train]
	test_data = shuffle_data[train:]
	basename ='' 
	if args.output is None:
		basename = args.input.split('.')[0]
	else:
		basename = args.output
	outputTrainTest([train_data,test_data],['train','test'],basename)
	

#double_check = 0
#for grouping,name in zip([perm1_train,perm1_val,perm1_test],['train','val','test']):
#    val = make_csvs(train_data, grouping,name,rec_to_pdb, pdb_to_ind)
#    print(float(val)/train_data.shape[0])
#    double_check += val
