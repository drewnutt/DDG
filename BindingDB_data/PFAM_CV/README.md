# Types files for Leave-One-Protein-Family-Out Cross Validation

There are two different types files:
 - `dg_<train|test>_pfam_cv_<#>.types` which only contains one protein, one ligand, and their binding affinity per line
 - `<train|test>_pfam_cv_<#>.types` which is used for &Delta;&Delta;G training and contains one protein and two ligands per line (the exact format is detailed below)

 The numbering of the types files corresponds to the left out protein family index. A mapping from the index to protein family name is provided in the previous directory as `pfam_values_map.csv`.

### Content
Each of the types files have lines of the form:  
    `<classification> <DDG> <DG_1> <DG_2> <Receptor file> <Ligand_1 file> <Ligand_2 file>`

`<classification>`, is 1 when the pK of the first ligand is greater than the pK of the second ligand and 0 otherwise.  
`<DDG>` is the difference in pK between the first ligand and the second ligand.  
`<DG_1>` and `<DG_2>` are the pKs of the first ligand and the second ligand, respectively. Subtracting these two values will give `<DDG>`.
