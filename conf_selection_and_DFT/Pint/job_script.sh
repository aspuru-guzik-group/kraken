#!/bin/bash
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=40
#SBATCH --time=20:00:00
#SBATCH --job-name Pint_calcs
#SBATCH --mail-type=ALL

# setup.sh
module load NiaEnv/2019b
module load python/3.6.8
export PATH="$PATH:$PROJECT/Pint/Multiwfn_3.7_bin_Linux_noGUI"
export PYTHONPATH="$PYTHONPATH:$PROJECT"

# Personal cleanup
rm  scratch_dir/* -rf

# Run the multiprocessing monster
CORES=800
python pint_main.py $CORES

