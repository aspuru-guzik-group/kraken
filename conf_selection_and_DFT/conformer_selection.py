#################################################
#                                               #
# Perform conformer selection of crest results  #
# to prepare for DFT runs.                      #
#                                               #
#################################################

from utils import *
import PL_conformer_selection_200411 as PLcs


if __name__ == '__main__':

	# Retrieve the list of ligands to run
	ligands = ligands_from_file('/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/example_ligands.txt')

	# Enter main kraken directory
	os.chdir(main_kraken_dir)
        # defined in utils.py

	# Run the conformer selection
	with Pool() as p:
		p.map(PLcs.conformer_selection_main, ligands)
