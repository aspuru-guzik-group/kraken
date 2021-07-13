#!/bin/bash
#SBATCH --partition=notchpeak
#SBATCH --account=sigman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --job-name monster_end.py
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N


module load gnu-parallel
module load gcc/8.3.0
module load gaussian16

module load anaconda3/2019.03
conda activate peters_PL1

#export PATH="/scratch/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"
#export crestpath=/home/a/aspuru/pascalf/codes/crest
#export PATH=$crestpath:$PATH
#export XTBPATH=/home/a/aspuru/pascalf/codes/xtb/build
#export PATH=$XTBPATH:$PATH
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80
export OMP_STACKSIZE=5G
export KMP_STACKSIZE=5G

ulimit -s unlimited

#export PYTHONPATH=/home/a/aspuru/pascalf/codes/morfeus:$PYTHONPATH
#export PATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PATH
#export PYTHONPATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PYTHONPATH

#export PYTHONPATH=/project/a/aspuru/passos/ligands_final_batch1_2020_02:$PYTHONPATH
export PYTHONPATH=/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use:$PYTHONPATH

#export PATH="/home/a/aspuru/pascalf/codes/chemaxon/marvinsuite/bin:$PATH"
#export PATH=$(pwd):$PATH
#export g16root="/project/a/aspuru/opt/gaussian"
#gr=$g16root
#export GAUSS_EXEDIR="$gr/g16C01/bsd:$gr/g16C01"
#export GAUSS_LEXEDIR="$gr/g16C01/linda-exe"
#export GAUSS_ARCHDIR="$gr/g16C01/arch"
#export GAUSS_BSDDIR="$gr/g16C01/bsd"
#export LD_LIBRARY_PATH="$GAUSS_EXEDIR:$LD_LIBRARY_PATH"
#export PATH="$PATH:$gr/gauopen:$GAUSS_EXEDIR"
#GAUSS_SCRDIR=$SCRATCH

#export PATH="$PATH:/project/a/aspuru/mlindn16/kraken/Pint/Multiwfn_3.7_bin_Linux_noGUI"
export Multifwnpath="/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/Pint/Multiwfn_3.7_bin_Linux_noGUI"
export PATH="$PATH:/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/Pint/Multiwfn_3.7_bin_Linux_noGUI"

export OMP_NUM_THREADS=1

#python ../python_scripts/run_end.py $1 $2
python /uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/run_end.py $1 $2
