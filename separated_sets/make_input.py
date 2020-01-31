import pandas as pd
import os
from itertools import permutations
import sys

def get_lig_id(rec,lig):
	lig_name = ''
	path = rec + '/' + lig
	with open(path, 'r') as f:
		for i,line in enumerate(f):
			if i == 1:
				lig_name = line[10:]
			if i > 1:
				break;
	try:
		name = float(lig_name)
	except:
		print(rec,lig)
	return float(lig_name)


full_list = pd.DataFrame()
affinity = pd.read_csv('ki_data_list.txt', sep='\t')
rec = affinity.iloc[:,0].unique().tolist()
rec_list = affinity['PDBID'].tolist()
rec.sort()
length = len(rec)
list_of_data = []
for i,r in enumerate(rec):
	is_rec = pd.Series([item==r for item in rec_list])
	affinity = affinity.assign(is_rec = is_rec.values)
	rec_affinity = affinity[affinity.is_rec == 1]
	path = r+'/'
	ligdict = {}
	ligs = os.listdir(path=path)
	r_name = ligs.pop()
	if len(ligs) < 2:
		continue;
	for lig in ligs:
		ligdict[lig]=get_lig_id(r,lig)
	lig_perms = permutations(ligs,2)
	for lig1,lig2 in lig_perms:
		l1 = rec_affinity.loc[rec_affinity['LigandMonomerID'] == ligdict[lig1]]
		l2 = rec_affinity.loc[rec_affinity['LigandMonomerID'] == ligdict[lig2]]
		try:
			label = int((float(l1['log(Ki)']) > float(l2['log(Ki)'])) == True)
		except:
			print(r,ligdict[lig1],ligdict[lig2])
			sys.exit();
		diff = float(l1['log(Ki)']) - float(l2['log(Ki)'])
		info = [label, diff, path+r_name , path+lig1, path+lig2]
		list_of_data.append(info)
	affinity.drop('is_rec', axis=1, inplace=True)
	if i%(length/10) == 0:
		print(i/(length/10))
output = pd.DataFrame(list_of_data)
output.to_csv('training_input.txt', sep=' ', index=False, header=False)
