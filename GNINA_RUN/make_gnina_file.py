import argparse
import csv

parser= argparse.ArgumentParser()
parser.add_argument('-f', help='input file for DG model')
parser.add_argument('-s','--start',default=4,type=int,help='column where the listing of receptor and ligands begins, 0-indexed')
parser.add_argument('-d','--dir',default='ddg_dataset',help='directory where all files are')
parser.add_argument('--model', help='gnina model to use')
args = parser.parse_args()
format_string = 'gnina -r {0}/{1} -l {0}/{2} --score_only --cnn {3} --cpu 1 > {4}.log'
last_lig = ''
with open(args.f, 'r') as f:
    filereader = csv.reader(f, delimiter=' ')
    for row in filereader:
        # if last_lig == row[args.start]:
        #         continue
        last_lig = row[args.start]
        rec = row[args.start].replace('_0.gninatypes','') + '.mol2'
        lig = row[args.start+1].replace('_0.gninatypes','') + '.mol2'
        logfile = args.model + '.' + lig.split('/')[-1].split('.')[0]
        print(format_string.format(args.dir, rec, lig, args.model,logfile))
