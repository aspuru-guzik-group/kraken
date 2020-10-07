# sterimol_db

Code to run sterimol calculations to create a database of ligands and their properties.



# Requirements
- xtb
- crest
- Modified version of Sterimol (https://github.com/bobbypaton/Sterimol), ask pascal.friederich@kit.edu
- Please add the path to your Sterimol code to the imports section of utils.py
- python 3.6, standard packages such as numpy, scipy, yaml, ...

# Usage
get_sterimol_parameters.py -smi "SMILES_CODE" -name molname  
or  
get_sterimol_parameters.py -xyz xyzfilename -name molname  

# Functionality
1.1 reads a xyz file in input_structures/xyzfilename.xyz  
or  
1.2 reads a smiles code and converts it to coordinates using rdkit. Saves the coordinates to input_structure/molname.xyz  
Then:  
2. optional: reads settings.yml (a sample settings.yml file is included in the code). If no settings.yml is given, default settings are used.  
3. runs crest  
4. extracts conformer geometries and energies etc.  
5. runs xtb on each conformer and extracts electronic structure information  
6. extracts the position of lone pairs and identifies the position where P binds to metal atoms  
7. runs sterimol on each conformer (incl. dummy atoms) and extracts sterimol parameters  
8. saves all results to results_all/molname.yml  






