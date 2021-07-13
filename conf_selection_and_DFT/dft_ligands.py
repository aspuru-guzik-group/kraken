################################
#                              #
# Submit DFT jobs for ligands. #
#                              #
################################

from utils import *
from dft_conformers import review_dft_conformer


def submit_dft_ligand(ligand: str) -> None:
	"""
	Run DFT computations for all conformers of a ligand.
	"""
	try:
		os.chdir(selected_confs_dir)
		os.chdir(ligand)

		# Load the job script lines
		fname = 'submit_gnuparallel.sh'
		with open(fname, 'r') as f:
			lines = f.readlines()

		# Edit the GNUPARALLEL script
		for i, line in enumerate(lines):

			# Update anaconda path
			if 'export PATH="/home/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"' in line:
				lines[i] = line.replace('home', 'scratch')

			# Update submission queue account
			if '#SBATCH --account' in line or '#SBATCH -A' in line:
				lines[i] = '#SBATCH -A ctb-ccgem\n'

		# Update file
		with open(fname, 'w') as f:
			for line in lines:
				f.write(line)

		# Submit jobs
		run(['chmod', '+x', 'run.py', 'end.py'])
		run(['sbatch', fname])
		os.chdir('..')

	except Exception as error:
		print(ligand)
		print(error)
		print()


def review_dft_ligand(ligand: str) -> None:
	"""
	Review the DFT computations for all conformers of a ligand.
	"""
	try:
		os.chdir(selected_confs_dir)

		# Determine conformers
		conformers = [file for file in os.listdir(ligand) if os.path.isdir(f"{ligand}/{file}") and 'backup' not in file]

		# Check for conformers
		if not conformers:
			raise ValueError('No conformers in ligand dir.')

		# Validate each
		for conformer in conformers:
			review_dft_conformer(conformer)

	except Exception as error:
		print(ligand)
		print(error)
		print()


if __name__ == '__main__':

	# Parse command line input
	try:
		mode = sys.argv[1]
		if mode not in ['--submit', '--review']:
			raise IndexError
	except IndexError:
		raise ValueError('Usage: "python dft_ligands.py --mode", where "mode" is either "submit" or "review".')

	# Retrieve the list of ligands to run
	ligands = ligands_from_file('../ligand_lists/ferrocenes.txt')

	# Enter selected conformers directory
	os.chdir(selected_confs_dir)

	# Unzip all the folders
	with Pool() as p:
		p.map(unzip, set(ligands))

	# Perform the main operation
	with Pool() as p:
		if mode == '--submit':
			p.map(submit_dft_ligand, ligands)
		elif mode == '--review':
			p.map(review_dft_ligand, ligands)
