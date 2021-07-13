##############################################
#                                            #
# Submit DFT jobs for individual conformers. #
#                                            #
##############################################

from utils import *

# Directory with Gaussian input files
com_path = Path('../conformer_lists/').resolve()

# Load the lines of the submit.sh file
with open('../bash_scripts/submit.sh') as f:
	submit_lines = f.readlines()


def submit_dft_conformer(conformer: str) -> None:
	"""
	Run DFT computation for a single conformer.
	"""
	try:
		os.chdir(selected_confs_dir)
		
		# Delete existing and enter fresh conf dir
		ligand = conformer.split('_')[0]
		
		#with suppress(FileNotFoundError):
		#	shutil.rmtree(f"{ligand}/{conformer}")
		#os.mkdir(f"{ligand}/{conformer}")
		os.chdir(f"{ligand}/{conformer}")

		# Create the submit.sh file
		with open('submit.sh', 'w') as f:
			for line in submit_lines:
				f.write(line.replace('LIGAND', ligand).replace('CONFORMER', conformer))
		
		# Delete existing log files if present
		#with suppress(FileNotFoundError):
		#	os.remove(f"{conformer}.log")

		# Import the gaussian input file
		print('TRANSFERING CONFORMER:', conformer)
		shutil.move(f"{com_path}/{conformer}.com", f"{conformer}.com")

		# Submit the job
		run(['sbatch', 'submit.sh'])

	except Exception as error:
		print(conformer)
		print(error)
		print()


def review_dft_conformer(conformer: str) -> None:
	"""
	Review DFT computation for a single conformer.
	"""
	try:
		os.chdir(selected_confs_dir)
		# Enter conformer dir
		ligand = conformer.split('_')[0]
		os.chdir(f"{ligand}/{conformer}")

		# Retrieve Gaussian output
		gauss_log = Path(f"{conformer}.log")
		if not gauss_log.exists():
			print(f"{conformer.ljust(19)} - NO GAUSSIAN LOG.")
			os.chdir('../..')
			return

		# Extract log contents
		with gauss_log.open('r') as f:
			lines = f.readlines()

		# Check for normal termination
		if 'Normal termination' in lines[-1]:
			print(f"{conformer.ljust(19)} - ok.")
		elif 'Error termination' in lines[-4] or 'Error termination' in lines[-5]:
			print(f"{conformer.ljust(19)} - ERROR TERMINATION.")
		else:
			print(f"{conformer.ljust(19)} - INSUFFUCIENT TIME.")
		os.chdir('../..')

	except Exception as error:
		print(conformer)
		print(error)
		print()


if __name__ == '__main__':

	# Parse command line input
	try:
		mode = sys.argv[1]
		if mode not in ['--submit', '--review']:
			raise IndexError
	except IndexError:
		raise ValueError('Usage: "python dft_conformers.py --mode", where "mode" is either "submit" or "review".')

	# Retrieve the list of conformers to run
	conformers = conformers_from_folder(com_path)[0:10]

	# Isolate the ligands
	ligands = ligands_from_conformers(conformers)

	# Enter selected conformers directory
	os.chdir(selected_confs_dir)

	# Unzip all the folders
	with Pool() as p:
		p.map(unzip, set(ligands))

	# Perform the main operation
	with Pool() as p:
		if mode == '--submit':
			p.map(submit_dft_conformer, conformers)
		elif mode == '--review':
			p.map(review_dft_conformer, conformers)
