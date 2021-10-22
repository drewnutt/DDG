import pandas as pd
import argparse
import numpy as np


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
    tt_comp = None
    while n_left:
        if len(ligs_left) == 0:
            break
        next_lig = ligs_left[rng.integers(len(ligs_left))]
        ligs_in.append(next_lig)
        tt_comp = cong_series[(cong_series['lig1'].isin(ligs_in)) & (cong_series['lig2'].isin(ligs_in))]
        n_left -= 1
        ligs_left.remove(next_lig)
    return tt_comp, n_left

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T','--train',required=True,help='Train types file')
    parser.add_argument('-E','--test',required=True,help='Test types file')
    parser.add_argument('-N',required=True,type=int,help='number of test ligands to include in the train file')
    parser.add_argument('-S','--random_seed',default=42,type=int,help='random seed')
    parser.add_argument('--test_subset',action='store_true',default=False,help='keep a subset of the test data, will only work if N=6')
    parser.add_argument('--stratify',action='store_true',default=False,help='Make a column that denotes the stratification of the two sets')
    args = parser.parse_args()

    failed = False
    rng = np.random.default_rng(seed=args.random_seed)

    og_test_data = pd.read_table(args.test, sep=' ', header=None)
    og_test_data.columns = ['cls','ddg', 'dg1','dg2','rec','lig1','lig2']
    test_data = og_test_data.copy()
    group_cts = test_data['rec'].value_counts()
    big_group = group_cts.index[0]
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
        while n_left > 0:
                series_set, n_left = get_more_ligs(rec,ligs,test_data,n_left,rng)
                tt_comp = tt_comp.append(series_set)
                test_data.drop(index=tt_comp.index,inplace=True,errors='ignore')
                if not len(test_data):
                        print(f'ran out of data, but still need {n_left} more ligands')
                        failed = True
                        break

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
                if not len(test_data):
                        print(f'ran out of data, but still need {n_left} more ligands')
                        failed = True
                        break
                rec = comp['rec'].unique().item()
                n_left -= 2

    if not failed and (args.N >= 6 and args.test_subset):
        leftover = og_test_data.drop(index=tt_comp.index)
        print('did not fail')
        if len(leftover):
            leftover.to_csv(f"{args.test.split('.')[0]}_TE.types", sep=' ' , header=False,index=False,float_format='%.4f')
    if args.stratify:
        train_data = pd.read_table(args.train, sep=' ', header=None)
        train_data['strat'] = 1
        train_data.to_csv(args.train,columns=['strat',1,2,3,4,5,6],sep=' ',header=False,index=False,float_format='%.4f') 
        tt_comp = tt_comp.copy()
        tt_comp['strat'] = 0
        tt_comp.to_csv(args.train, columns=['strat','ddg', 'dg1','dg2','rec','lig1','lig2'], sep=' ', mode='a', header=False,index=False,float_format='%.4f') 
    else:
        tt_comp.to_csv(args.train, columns=['cls','ddg', 'dg1','dg2','rec','lig1','lig2'], sep=' ',header=False,index=False,mode='a',float_format='%.4f') 
