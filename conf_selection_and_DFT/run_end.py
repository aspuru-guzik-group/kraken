###################################
#                                 #
# Parse completed DFT properties. #
#                                 #
###################################

import numpy as np

from utils import *

# Save the path with the new end.py script
tobi_new_path = Path('/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/').resolve()


def run_end(ligand: str) -> None:
	"""
	Run the end.py script for a given ligand.
	"""
	# Enter selected confs dir
	os.chdir(selected_confs_dir)

	print(f"MLD_LOGGER, fname=run_end.py, ligand={ligand}: Pre end.py.")
	try:
		# Unzip the file, if necessary
		unzip(ligand)

		# Enter ligand dir and copy new end.py script
		os.chdir(ligand)
		shutil.copyfile(f"{tobi_new_path}/tobi_new_end.py", 'end.py')

		# Run end.py
		run(['python', 'end.py'])
		print(f"MLD_LOGGER, fname=run_end.py, ligand={ligand}: Post end.py, no errors.")	

	except Exception as error:
		print(ligand)
		print(error)
		print(f"MLD_LOGGER, fname=run_end.py, ligand={ligand}: Post end.py, yes errors.")


if __name__ == '__main__':

	if len(sys.argv) == 2:

		# Use the only input as the ligand of choice
		ligands = sys.argv[1:]

	elif len(sys.argv) == 3:

		# Retrieve the ID of the job and number of batches
		job_id = int(sys.argv[1])
		batches = int(sys.argv[2])

		# All ligands
		all_ligands = np.asarray(sorted(set([file.split('.')[0] for file in os.listdir(selected_confs_dir) if file[0:4] == '0000'])))

		# Partition into subset
		ligands = list(np.array_split(all_ligands, batches)[job_id])

	else:
		ligands = ligands_from_file('/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/example_ligands.txt')

	# Run over every ligand in parallel
	#nproc = max((os.cpu_count() - 2, 1))
	#with Pool(nproc) as p:
		#p.map(run_end, ligands)

	for ligand in ligands:
		run_end(ligand)

