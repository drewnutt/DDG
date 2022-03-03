# Data for Jimenez-Luna Comparison

This data is a subset of the data available from the BindindDB 3D Structure Series Dataset. We take our filtered dataset and remove any ligands without an IC<sub>50</sub> measurement. Then we cluster the congeneric series via 90% sequence similarity of the protein target and remove any duplicate ligands within the cluster.

### Content
Each of the types files have lines of the form:  
    `<classification> <DDG> <DG_1> <DG_2> <Receptor file> <Ligand_1 file> <Ligand_2 file>`

`<classification>`, is 1 when the pK of the first ligand is greater than the pK of the second ligand and 0 otherwise.  
`<DDG>` is the difference in pK between the first ligand and the second ligand.  
`<DG_1>` and `<DG_2>` are the pKs of the first ligand and the second ligand, respectively. Subtracting these two values will give `<DDG>`.
