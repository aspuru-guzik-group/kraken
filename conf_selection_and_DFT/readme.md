The full kraken workflow involves three steps. This folder contains scripts and information for steps 2 and 3. 
# Step 1. CREST conformer search + xTB calculations
	- see folder for instructions and information

# Step 2. conformer selection + DFT calculations
	- input: results (ymls) from XTB workflow portion
	- recommended usage
		- edit conformer_selection.py to refer to example_ligands.txt
		- replace sub16_PL with equivalent submission script
		- submit job as 'sbatch conf_script.sh' or equivalent

# Step 3. gather and condense properties 
	- input: results from conformer selection + DFT calculations workflow portion
	- recommended usage
		- submit job as 'sbatch end_script.sh' or equivalent 

	
# Requirements
- xTB
- crest
- Gaussian
- openbabel
- python 3.6
- morfeus
- rdkit
- cython
- numpy, scipy, yaml, seaborn, time, r-uuid, r-getpass, pathlib, cclib, pyvista, vtk, sympy, tqdm, dataclasses, fire, joblib


note: v0 scripts contain user-specific paths. these will be condensed in a future update. path locations are listed below. 
- ded.py							              lines 13, 14
- pint_main.py						          lines 36, 63
- conf_script.sh 					          lines 28, 34
- conformer_selection.py		      	line  15
- utils.py							            lines 20, 23, 26
- run_end.py						            lines 12, 61
- end_script.sh						          lines 37, 52, 53, 58
- PL_conformer_selection_200411.py	lines 834, 836, 
