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
        self.caff
    def get_ligand(self):
        return "{} {} {} {}".format(self.name,self.aff,self.caff,self.score)

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
gnina_df = 'gnina_out_{}.txt'.format(args.model)
with open(gnina_df,'w') as df:
    for lig in ligand_list:
        df.write(lig.get_ligand())
del ligand_list

full_list = pd.DataFrame()
affinity = pd.read_csv(gnina_df, sep=' ', header=None, names=['name','caff','score','aff'])
rec = affinity['name'].str[:4].unique().tolist()
rec_list = affinity['name'].iloc[:,0].tolist()
rec.sort()
length = len(rec)
list_of_data = []
for i,r in enumerate(rec):
	is_rec = pd.Series([item==r for item in rec_list])
	affinity = affinity.assign(is_rec = is_rec.values)
	rec_affinity = affinity[affinity.is_rec == 1]
	ligdict = {}
	ligs = os.listdir(path=path)
        ligs= [item[:6] for item in ligs if '.mol2' in item and '_' not in item]
	if len(ligs) < 2:
		continue;
        ligs.sort()
	lig_perms = permutations(ligs,2)
	for lig1,lig2 in lig_perms:
		try:
			labelcaff = int((float(lig1['caff']) > float(lig2['caff'])) == True)
			labelscr = int((float(lig1['score']) > float(lig2['score'])) == True)
			labelv = int((float(lig1['aff']) > float(lig2['aff'])) == True)
		except:
			print(r,lig1,lig2)
		diffcaff = float(lig1['caff']) - float(lig2['caff'])
		diffscore = float(lig1['score']) - float(lig2['score'])
		diffv = float(lig1['aff']) - float(lig2['aff'])
		info = [labelcaff, diffcaff, labelscr, diffscore, labelv, diffv, lig1,lig2]
		list_of_data.append(info)
	affinity.drop('is_rec', axis=1, inplace=True)
	if i%(length/10) == 0:
		print(i/(length/10))
output = pd.DataFrame(list_of_data)
output.columns = ['labelcaff', 'diffcaff', 'labelscr', 'diffscr', 'labelvina','diffvina', 'lig1', 'lig2']
gnina_ddg = 'gnina_DDG_{}.txt'.format(args.model)
output.to_csv(gnina_ddg, sep=' ', index=False)

train_data = pd.read_csv('training_input.txt', sep=' ', header=None, names=['label','diff','lig1','lig2'], usecols=[0,1,3,4])
train_data['lig2'] = train_data['lig2'].apply(lambda x: x[5:11])
train_data['lig1'] = train_data['lig1'].apply(lambda x: x[5:11])
assert gnina_data.shape[0] == train_data.shape[0]
gnina_lbls = gnina_ddg[['labelcaff','labelscr','labelvina']].copy()
labels = train_data['label']
comp_dict = gnina_lbls.eq(labels, axis=0).mean(axis=0, numeric_only=True).to_dict()
print('{}:\nMetric | Accuracy\n-----|-----\nGNINA Affinity | {}\n GNINA Score | {}'.format(args.model, comp_dict['labelcaff'],comp_dict['labelscr']))
if args.vina:
    print('Vina Affinity | {}'.format(comp_dict['labelvina'])
