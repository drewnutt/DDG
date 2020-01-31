#!/bin/bash
# Roshni Bhatt
# make directory for each receptor PDB ID and fill with split ligand files. One file (1UWJ) is empty; cannot be found on site.

for line in $(cat results_list2.txt)
do
	n=$(grep '<TRIPOS>MOLECULE' ../$line | wc -l)
	((n=n-2))
        pdb=$(echo $line | sed 's/-results.mol2//g')
	mkdir $pdb
	cd $pdb
	csplit -f $pdb ../../$line '/<TRIPOS>MOLECULE/' {$n}
	cd ../
done
