# kraken
Code to compute electronic and steric features to create a database of ligands and their properties.

# Requirements
- xtb
- crest
- morfeus (like will be added soon)
- python 3.6, standard packages such as numpy, scipy, yaml, ...

# Recommended usage
- copy and adjust one of the yml files in example_settings to your working directory and name it settings.yml
- generate a csv file with your smiles codes ("input.yml")
- use the following command:
run_kraken.py -idx {} input.csv Ni
or
run_kraken.py -idx {} input.csv noNi

# 3D conversion
- if you only have smiles codes of the ligands, please use any of the following "3D_conversion_flag" in your input.csv file:
-- 0: rdkit will be used to convert the smiles code to 3D
-- 1: molconvert (ChemAxon) will be used to convert the smiles code to 3D
-- 3: obabel will be used to convert the smiles code to 3D
-- 4: all aforementioned methods will be tried to convert the smiles code to 3D
- if you already have 3D coordinate files in xyz format, you can use the following "3D_conversion_flag" in your input.csv file:
-- 2: searches for a file input_structures_Ni/#ID.xyz or input_structures_noNi/#ID.xyz

# Functionality
1.1 reads a xyz file in input_structures/xyzfilename.xyz  
or  
1.2 reads a smiles code and converts it to coordinates using the preferred 3D conversion method. Saves the coordinates to input_structure/molname.xyz  
Then:  
2. optional: reads settings.yml (a sample settings.yml file is included in the code). If no settings.yml is given, default settings are used.  
3. runs crest  
4. extracts conformer geometries and energies etc.  
5. runs xtb on each conformer and extracts electronic structure information  
6. extracts the position of lone pairs and identifies the position where P binds to metal atoms  
7. runs morfeus on each conformer (incl. dummy atoms) and extract additional features  
8. saves all results to results_all_Ni/#ID.yml or results_all_noNi/#ID.yml  






