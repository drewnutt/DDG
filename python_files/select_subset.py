import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-T','--train',required=True,help='Train types file')
parser.add_argument('-E','--test',required=True,help='Test types file')
parser.add_argument('-N',required=True,type=int,help='number of test ligands to include in the train file')
parser.add_argument('-S','--random_seed',default=42,type=int,help='random seed')
parser.add_argument('--test_subset',action='store_true',default=False,help='keep a subset of the test data, will only work if N=6')
args = parser.parse_args()

def split_comparison(comparison,rng):
    comparison.at[:,'cls'] = -1
    comparison.at[:,'ddg'] = 0
    vals = ['dg1','lig1']
    if rng.random() > 0.5:
        vals = ['dg2','lig2']
    comparison.at[:,vals[0]] = 0
    comparison.at[:,vals[1]] = 'None'

    return comparison

def get_more_ligs(rec,ligs_in,test_data,n_left,rng):
    cong_series = test_data[test_data['rec'] == rec]
    ligs_left = set(cong_series['lig1'].unique().tolist() + cong_series['lig2'].unique().tolist()) - set(ligs_in)
    ligs_left = sorted(list(ligs_left))
    while n_left:
        if len(ligs_left) == 0:
            break
        next_lig = ligs_left[rng.integers(len(ligs_left))]
        ligs_in.append(next_lig)
        tt_comp = cong_series[(cong_series['lig1'].isin(ligs_in)) & (cong_series['lig2'].isin(ligs_in))]
        n_left -= 1
        ligs_left.remove(next_lig)
    return tt_comp, n_left

    
rng = np.random.default_rng(seed=args.random_seed)

test_data = pd.read_table(args.test, sep=' ', header=None)
test_data.columns = ['cls','ddg', 'dg1','dg2','rec','lig1','lig2']
group_cts = test_data.value_counts(subset=['rec'])
big_group = group_cts.index[0][0]
num_ligs = (1+np.sqrt(1+4*group_cts[0]))/2
print(f"{big_group}: {num_ligs}")
series = test_data[test_data['rec'] == big_group].copy()
tt_comp = series.sample(n=1, random_state=args.random_seed)
n_left = args.N-2
if args.N == 1:
    tt_comp = split_comparison(tt_comp,rng)
else:
    ligs = [tt_comp['lig1'].item(), tt_comp['lig2'].item()]
    tt_comp = test_data[(test_data['lig1'].isin(ligs)) & (test_data['lig2'].isin(ligs))]
    test_data.drop(index=tt_comp.index,inplace=True)
    rec = tt_comp['rec'].unique().item()
    if n_left:
        while n_left > 0:
            series_set, n_left = get_more_ligs(rec,ligs,test_data,n_left,rng)
            tt_comp = tt_comp.append(series_set)
            test_data.drop(index=tt_comp.index,inplace=True,errors='ignore')

            if n_left == 0:
                break
            comp = test_data.sample(n=1,random_state=args.random_seed)
            if n_left == 1:
                comp = split_comparison(comp,rng)
                tt_comp = tt_comp.append(comp)
                break
            ligs = [comp['lig1'].item(), comp['lig2'].item()]
            comp = test_data[(test_data['lig1'].isin(ligs)) & (test_data['lig2'].isin(ligs))]
            tt_comp = tt_comp.append(comp)
            test_data.drop(index=comp.index,inplace=True)
            rec = comp['rec'].unique().item()
            n_left -= 2

if args.N ==6 and args.test_subset:
    leftover = series.drop(index=tt_comp.index)
    if leftover is not None:
        leftover.to_csv(f"{args.test.split('.')[0]}_TE.types", sep=' ' , header=False,index=False)
tt_comp.to_csv(args.train,sep=' ',header=False,index=False,mode='a') 
