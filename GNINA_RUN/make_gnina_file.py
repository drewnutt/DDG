import argparse
import csv

parser= argparse.ArgumentParser()
parser.add_argument('-f', help='input file for DDG model')
parser.add_argument('--model', help='gnina model to use')
args = parser.parse_args()
format_string = 'gnina -r separated_sets/{} -l separated_sets/{} --score_only --gpu --cnn_scoring --cnn {}'
last_lig = ''
with open(args.f, 'r') as f:
    filereader = csv.reader(f, delimiter=' ')
    for row in filereader:
        if last_lig == row[3]:
                continue
        last_lig = row[3]
        rec = row[2][:11] + '.mol2'
        lig = row[3][:11] + '.mol2'
        print(format_string.format(rec, lig,args.model))
