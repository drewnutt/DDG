import pandas as pd
import argparse


parser= argparse.ArgumentParser()
parser.add_argument('--trainf', required=True, help='training input file for DDG model')
parser.add_argument('--model', required=True, help='gnina model used to calculate info')
args = parser.parse_args()

gnina_ddg = 'gnina_DDG_{}.txt'.format(args.model)
ensemble = pd.read_csv(gnina_ddg, sep=' ')
for i in range(4):
    j = i + 1
    gnina_ddg = 'gnina_DDG_{}_{}.txt'.format(args.model, j)
    model = pd.read_csv(gnina_ddg, sep=' ')
    ensemble['l_gnina_aff'] = ensemble['l_gnina_aff'] + model['l_gnina_aff']
    ensemble['l_scr'] = ensemble['l_scr'] + model['l_scr']

ensemble['l_gnina_aff'] = ensemble['l_gnina_aff']/5.0
ensemble['l_scr'] = ensemble['l_scr']/5.0
ensemble['l_gnina_aff'] = ensemble['l_gnina_aff'].round()
ensemble['l_scr'] = ensemble['l_scr'].round()
train_data = pd.read_csv(args.trainf, sep=' ', header=None, names=['label','diff','lig1','lig2'], usecols=[0,1,3,4])
train_data['lig2'] = train_data['lig2'].apply(lambda x: x[5:11])
train_data['lig1'] = train_data['lig1'].apply(lambda x: x[5:11])
assert ensemble.shape[0] == train_data.shape[0]
gnina_lbls = ensemble[['l_gnina_aff','l_scr']].copy()
labels = train_data['label']
comp_dict = gnina_lbls.eq(labels, axis=0).mean(axis=0, numeric_only=True).to_dict()
output_string = '{}:\nMetric | Accuracy\n-----|-----\nGNINA Affinity | {:.4f}\nGNINA Score | {:.4f}'.format(args.model, comp_dict['l_gnina_aff'],comp_dict['l_scr'])
model_stat = '{}_{}_stats.md'.format('ensemble',args.model)
with open(model_stat,'w') as f:
    f.write(output_string)
