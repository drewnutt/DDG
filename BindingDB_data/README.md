# Data for &Delta;&Delta;G prediction

All of the data was pulled from the [BindingDB congeneric series Dataset](http://www.bindingdb.org/bind/surflex_entry.jsp).

This directory contains the following files:
 - `BuildNewBDBSet.ipynb` - used to filter the dataset and create the types files.
 - `all_data_t*_papersplit_rand*_*_*.types` - additional ligands dataset types files
 - `all_newdata.types` - types file containing all of the 2-permutations of all of the data
 - `dg_all_newdata.types` - types file containing all of the protein-ligand pairs in all of the data with affinity values
 - `fullbdb_*.txt` - all of the information for the protein-ligand pairs with measurements of the type given in the filename
 - `pfam_addtrain_nums.txt`
 - `pfam_values_map.txt`

## Additional Ligands Types files
### Naming convention
All of the types files have names of the form:  

    `all_data_<train|test>_papersplit_rand<#>_<p|c>_<# additional ligands>.types`
where the `p` or `c` denotes whether the file contains 2-permutations or 2-combinations, respectively, of the ligands.

### Content
Each of the types files have lines of the form:  
    `<classification> <DDG> <DG_1> <DG_2> <Receptor file> <Ligand_1 file> <Ligand_2 file>`

`<classification>`, is 1 when the pK of the first ligand is greater than the pK of the second ligand and 0 otherwise.  
`<DDG>` is the difference in pK between the first ligand and the second ligand.  
`<DG_1>` and `<DG_2>` are the pKs of the first ligand and the second ligand, respectively. Subtracting these two values will give `<DDG>`.
