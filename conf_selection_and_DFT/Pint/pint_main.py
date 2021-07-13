###############################################
#                                             #
# Compute the Pint parameters for a database  #
# of phospines with conformers.               #
#                                             #
# Usage:                                      #
# 	python main.py (CORES)                #
#                                             #
# CORES: Optional integer                     #
#	- Number of cores for multiprocessing #
# 	- If unspecified, serial job          #
#                                             #
###############################################

# Remainder of the imports
import os
import sys
import shutil
import os.path
from glob import glob
from multiprocessing import Pool
from subprocess import run, DEVNULL


def compute_pint(phos_idx: str) -> None:
	"""
	Compute the P_int parameters for a given phosphine.
	The calculation is run across all conformers and stored
	in a .csv file for all provided .xyz's and .fchk's.

	Args:
		phos_idx: ID of phosphine, e.g. 00000127
	"""

	# List the data paths
	data_path = '/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/selected_conformers'
	data_dir  = f"{data_path}/{phos_idx}"
	data_zip  = f"{data_dir}.zip"	

	# Check for unzipped results
	if os.path.idsir(data_dir):
		pass
	
	# Check for zipped results
	elif os.path.isfile(data_zip):
		run(['unzip', data_zip], stdout=DEVNULL)			
	
	# Nothing exists
	else: 
		return

	# Gather all the conformer paths
	org_dirs   = glob(f"{data_dir}/{phos_idx}_noNi_*")		
	inorg_dirs = glob(f"{data_dir}/{phos_idx}_Ni_*")
	conf_dirs = org_dirs + inorg_dirs
	conf_dirs.sort()

	# Run through all conformers
	for conf_dir in conf_dirs:

		# Step in and run the Pint code
		os.chdir(conf_dir)
		run(['python', '/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/Pint/P_int.py'], stdout=DEVNULL)
		os.chdir('..')


if __name__ == '__main__':

	# Scan all different phosphines
	phos_indices = [idx.zfill(8) for idx in range(1, 2004)]

	# Use multiprocessing Pool
	try:
		cores = int(sys.argv[1])
		print('Using multiprocessing.')
		with Pool(cores) as p:
			p.map(compute_pint, phos_indices)

	# Synchronous
	except:	
		print('Using sequential processing.')
		for phos_idx in phos_indices:	
			compute_pint(phos_idx)

