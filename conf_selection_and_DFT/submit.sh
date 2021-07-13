#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:55:00
#SBATCH --job-name final_subs
#SBATCH -A rrg-aspuru

module load NiaEnv/2018a

module load intel/2017.7 intelmpi/2017.7 
module load java/1.8.0_162
export PATH="/scratch/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"
export crestpath=/home/a/aspuru/pascalf/codes/crest
export PATH=$crestpath:$PATH
export XTBPATH=/home/a/aspuru/pascalf/codes/xtb/build
export PATH=$XTBPATH:$PATH
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80
export OMP_STACKSIZE=5G
ulimit -s unlimited
export PYTHONPATH=/home/a/aspuru/pascalf/codes/morfeus:$PYTHONPATH
export PATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PATH
export PYTHONPATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PYTHONPATH
export PATH="/home/a/aspuru/pascalf/codes/chemaxon/marvinsuite/bin:$PATH"
export g16root="/project/a/aspuru/opt/gaussian"
gr=$g16root
export GAUSS_EXEDIR="$gr/g16C01/bsd:$gr/g16C01"
export GAUSS_LEXEDIR="$gr/g16C01/linda-exe"
export GAUSS_ARCHDIR="$gr/g16C01/arch"
export GAUSS_BSDDIR="$gr/g16C01/bsd"
export LD_LIBRARY_PATH="$GAUSS_EXEDIR:$LD_LIBRARY_PATH"
export PATH="$PATH:$gr/gauopen:$GAUSS_EXEDIR"
GAUSS_SCRDIR=$SCRATCH

g16 < /gpfs/fs0/project/a/aspuru/passos/ligands_final_batch1_2020_02/selected_conformers/LIGAND/CONFORMER/CONFORMER.com > /gpfs/fs0/project/a/aspuru/passos/ligands_final_batch1_2020_02/selected_conformers/LIGAND/CONFORMER/CONFORMER.log
formchk /gpfs/fs0/project/a/aspuru/passos/ligands_final_batch1_2020_02/selected_conformers/LIGAND/CONFORMER/CONFORMER.chk /gpfs/fs0/project/a/aspuru/passos/ligands_final_batch1_2020_02/selected_conformers/LIGAND/CONFORMER/CONFORMER.fchk
vmin4.py CONFORMER.fchk   ##changed from vmin3.py by EP 05/17/2021

