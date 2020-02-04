import argparse
import csv
import pandas as pd
from itertools import permutations
import os
import sys

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
args = parser.parse_args()

ligand_list = []
with open(args.gf, 'r') as f:
    name = ''
    aff, caff, score=0,0,0
    for idx, line in enumerate(f):
        if idx%5==0:
            name = line[64:70]
        if idx%5==1:
            aff = float(line[10:18])
        if idx%5==2:
            score= float(line[10:])
        if idx%5==3:
            caff=float(line[12:])
        if idx%5==4:
            ligand_list.append(Ligand(name,caff,score,aff))
ligand_list.sort(key=lambda x: x.name)
gnina_df = 'GNINA_RUN/gnina_out_{}.txt'.format(args.model)
with open(gnina_df,'w') as df:
    for lig in ligand_list:
        df.write(lig.get_ligand())
del ligand_list

full_list = pd.DataFrame()
affinity = pd.read_csv(gnina_df, sep=' ', header=None, names=['name','caff','score','aff'])
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
            labelcaff = int(abs((float(rec_affinity.at[lig1,'caff'])) > abs(float(rec_affinity.at[lig2,'caff']))) == True)
            labelscr = int(abs((float(rec_affinity.at[lig1,'score'])) > abs(float(rec_affinity.at[lig2,'score']))) == True)
            labelv = int(abs((float(rec_affinity.at[lig1,'aff'])) > abs(float(rec_affinity.at[lig2,'aff']))) == True)
            diffcaff = abs(float(rec_affinity.at[lig1,'caff'])) - abs(float(rec_affinity.at[lig2,'caff']))
            diffscore = abs(float(rec_affinity.at[lig1,'score'])) - abs(float(rec_affinity.at[lig2,'score']))
            diffv = abs(float(rec_affinity.at[lig1,'aff'])) - abs(float(rec_affinity.at[lig2,'aff']))
            info = [labelcaff, diffcaff, labelscr, diffscore, labelv, diffv, lig1,lig2]
            list_of_data.append(info)
        except:
            print(r,lig1,lig2)
    affinity.drop('is_rec', axis=1, inplace=True)
    if i%(length/10) == 0:
        print(i/(length/10))
output = pd.DataFrame(list_of_data)
output.columns = ['labelcaff', 'diffcaff', 'labelscr', 'diffscr', 'labelvina','diffvina', 'lig1', 'lig2']
gnina_ddg = 'GNINA_RUN/gnina_DDG_{}.txt'.format(args.model)
output.to_csv(gnina_ddg, sep=' ', index=False)

train_data = pd.read_csv(args.trainf, sep=' ', header=None, names=['label','diff','lig1','lig2'], usecols=[0,1,3,4])
train_data['lig2'] = train_data['lig2'].apply(lambda x: x[5:11])
train_data['lig1'] = train_data['lig1'].apply(lambda x: x[5:11])
assert output.shape[0] == train_data.shape[0]
gnina_lbls = output[['labelcaff','labelscr','labelvina']].copy()
labels = train_data['label']
comp_dict = gnina_lbls.eq(labels, axis=0).mean(axis=0, numeric_only=True).to_dict()
output_string = '{}:\nMetric | Accuracy\n-----|-----\nGNINA Affinity | {}\nGNINA Score | {}'.format(args.model, comp_dict['labelcaff'],comp_dict['labelscr'])
if args.vina:
    output_string += 'Vina Affinity | {}'.format(comp_dict['labelvina'])
model_stat = 'GNINA_RUN/{}_stats.txt'.format(args.model)
with open(model_stat,'w') as f:
    f.write(output_string)
