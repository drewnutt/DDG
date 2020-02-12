import argparse
import re
import csv
import pandas as pd
from itertools import permutations
import os
import sys
import numpy as np

class Ligand():
    def __init__(self,name,caff,score,aff):
        self.name=name
        self.aff=aff
        self.score=score
        self.caff=caff
    def get_ligand(self):
        return "{} {} {} {}\n".format(self.name,self.caff,self.score,self.aff)

parser= argparse.ArgumentParser()
parser.add_argument('-gf', required=True, help='file containing condensed gnina output')
parser.add_argument('--trainf', required=True, help='training input file for DDG model')
parser.add_argument('--vina', action='store_true', help='report classification using Vina affinity')
parser.add_argument('--model', help='gnina model used to calculate info')
parser.add_argument('--cv_set', action='store_true', help='check accuracy on subset of all data')
args = parser.parse_args()

ligand_list = []
name_p = re.compile('(?:/)([A-Z0-9]+)(?:\.mol2)')
with open(args.gf, 'r') as f:
    name = ''
    aff, caff, score=0,0,0
    for idx, line in enumerate(f):
        vals = line.split()
        if idx%5==0:
            name = re.findall(name_p,vals[4])[0]
        if idx%5==1:
            aff = vals[1]
        if idx%5==2:
            score= vals[1]
        if idx%5==3:
            caff= vals[1]
        if idx%5==4:
            ligand_list.append(Ligand(name,caff,score,aff))
ligand_list.sort(key=lambda x: x.name)
gnina_df = 'GNINA_RUN/gnina_out_{}.txt'.format(args.model)
with open(gnina_df,'w') as df:
    for lig in ligand_list:
        df.write(lig.get_ligand())
del ligand_list

full_list = pd.DataFrame()
affinity = pd.read_csv(gnina_df, sep=' ', header=None, names=['name','gnina_aff','score','vaff'], dtype={'name':str,'gnina_aff':np.float64,'score':np.float64,'vaff':np.float64})
rec = affinity['name'].str[:4].unique().tolist()
rec_list = affinity['name'].str[:4].tolist()
rec.sort()
length = len(rec)
list_of_data = []
for i,r in enumerate(rec):
    is_rec = pd.Series([item==r for item in rec_list])
    affinity = affinity.assign(is_rec = is_rec.values)
    rec_affinity = affinity[affinity.is_rec == 1]
    rec_affinity.set_index('name', inplace=True)
    ligdict = {}
    ligs = os.listdir(path='separated_sets/'+ r+'/')
    ligs= [item[:6] for item in ligs if '.mol2' in item and '_' not in item]
    if len(ligs) < 2:
        continue;
    ligs.sort()
    lig_perms = permutations(ligs,2)
    for lig1,lig2 in lig_perms:
        try:
            labelgaff = int(rec_affinity.at[lig1,'gnina_aff'] < rec_affinity.at[lig2,'gnina_aff']) #1 if 2nd ligand has higher affinity(0 if 1st ligand has higher affinity)
            labelscr = int(rec_affinity.at[lig1,'score'] < rec_affinity.at[lig2,'score'])
            labelv = int(rec_affinity.at[lig1,'vaff'] > rec_affinity.at[lig2,'vaff'])
            diffgaff = rec_affinity.at[lig1,'gnina_aff'] - rec_affinity.at[lig2,'gnina_aff']
            diffscore = rec_affinity.at[lig1,'score'] - rec_affinity.at[lig2,'score']
            diffv = rec_affinity.at[lig1,'vaff'] - rec_affinity.at[lig2,'vaff']
            info = [labelgaff, diffgaff, labelscr, diffscore, labelv, diffv, lig1,lig2]
            list_of_data.append(info)
        except Exception as e:
            print(r,lig1,lig2, e)
    affinity.drop('is_rec', axis=1, inplace=True)
    if i%(length/10) == 0:
        print(i/(length/10))
output = pd.DataFrame(list_of_data)
output.columns = ['l_gnina_aff', 'd_gnina_aff', 'l_scr', 'd_scr', 'l_vina_aff','d_vina_aff', 'lig1', 'lig2']
gnina_ddg = 'GNINA_RUN/gnina_DDG_{}.txt'.format(args.model)
output.to_csv(gnina_ddg, sep=' ', index=False)

train_data = pd.read_csv(args.trainf, sep=' ', header=None, names=['label','diff','lig1','lig2'], usecols=[0,1,3,4], dtype={'label':np.int32,'diff':np.float64})
ligname_pattern = re.compile('(?:/)(......)(?:_.\.gninatypes)')
recname_pattern = re.compile('(....)')
train_data['lig2'] = train_data['lig2'].apply(lambda x: re.findall(ligname_pattern,x)[0])
train_data['lig1'] = train_data['lig1'].apply(lambda x: re.findall(ligname_pattern,x)[0])
outfile_addn = ''
if args.cv_set:
    u_train_r = train_data['lig1'].str[:4].unique().tolist()
    gnina_r = output['lig1'].str[:4].tolist()
    is_train = pd.Series([item in u_train_r for item in gnina_r])
        
    output = output.assign(is_train = is_train.values)
    output = output[output.is_train == 1]
    outfile_addn = 'cv_set_'
    output.sort_values(by=['lig1','lig2'],inplace=True, ignore_index=True)
    train_data.sort_values(by=['lig1','lig2'],inplace=True, ignore_index=True)
assert output.shape[0] == train_data.shape[0]
print(output.head(10),train_data.head(10))
gnina_lbls = output[['l_gnina_aff','l_scr','l_vina_aff']].copy()
labels = train_data['label']
comp_dict = gnina_lbls.eq(labels, axis=0).mean(axis=0, numeric_only=True).to_dict()
output_string = '# {}:\nMetric | Accuracy\n-----|-----\nGNINA Affinity | {:.4f}\nGNINA Score | {:.4f}'.format(args.model, comp_dict['l_gnina_aff'],comp_dict['l_scr'])
if args.vina:
    output_string += '\nVina Affinity | {:.4f}'.format(comp_dict['l_vina_aff'])
model_stat = 'GNINA_RUN/{}{}_stats.md'.format(outfile_addn, args.model)
with open(model_stat,'w') as f:
    f.write(output_string)
