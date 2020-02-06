import pandas as pd
import argparse


parser= argparse.ArgumentParser()
parser.add_argument('--trainf', required=True, help='training input file for DDG model')
parser.add_argument('--vina', action='store_true', help='report classification using Vina affinity')
parser.add_argument('--model', required=True, help='gnina model used to calculate info')
args = parser.parse_args()

gnina_ddg = 'gnina_DDG_{}.txt'.format(args.model)
ensemble = pd.read_csv(gnina_ddg, sep=' ')
for i in range(4):
    j = i + 1
    gnina_ddg = 'gnina_DDG_{}_{}.txt'.format(args.model, j)
    model = pd.read_csv(gnina_ddg, sep=' ')
    ensemble['labelcaff'] = ensemble['labelcaff'] + model['labelcaff']
    ensemble['labelscr'] = ensemble['labelscr'] + model['labelscr']

ensemble['labelcaff'] = ensemble['labelcaff']/5.0
ensemble['labelscr'] = ensemble['labelscr']/5.0
ensemble['labelcaff'] = ensemble['labelcaff'].round()
ensemble['labelscr'] = ensemble['labelscr'].round()
train_data = pd.read_csv(args.trainf, sep=' ', header=None, names=['label','diff','lig1','lig2'], usecols=[0,1,3,4])
train_data['lig2'] = train_data['lig2'].apply(lambda x: x[5:11])
train_data['lig1'] = train_data['lig1'].apply(lambda x: x[5:11])
assert ensemble.shape[0] == train_data.shape[0]
gnina_lbls = ensemble[['labelcaff','labelscr','labelvina']].copy()
labels = train_data['label']
comp_dict = gnina_lbls.eq(labels, axis=0).mean(axis=0, numeric_only=True).to_dict()
output_string = '{}:\nMetric | Accuracy\n-----|-----\nGNINA Affinity | {}\nGNINA Score | {}'.format(args.model, comp_dict['labelcaff'],comp_dict['labelscr'])
if args.vina:
    output_string += 'Vina Affinity | {}'.format(comp_dict['labelvina'])
model_stat = '{}_{}_stats.txt'.format('ensemble',args.model)
with open(model_stat,'w') as f:
    f.write(output_string)
