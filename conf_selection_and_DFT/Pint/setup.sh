#!/bin/bash

module load NiaEnv/2019b
module load python/3.6.8
export PATH="$PATH:/project/a/aspuru/mlindn16/Pint/Multiwfn_3.7_bin_Linux_noGUI"
export PYTHONPATH="$PYTHONPATH:$PROJECT"



module load gcc/8.3.0

export PATH="/scratch/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"

# crest
export crestpath=/home/a/aspuru/pascalf/codes/crest
export PATH=$crestpath:$PATH

# XTB
export XTBPATH=/home/a/aspuru/pascalf/codes/xtb/build
export PATH=$XTBPATH:$PATH
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80
export OMP_STACKSIZE=5G

ulimit -s unlimited

# own codes
export PYTHONPATH=/home/a/aspuru/pascalf/codes/morfeus:$PYTHONPATH
export PATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PATH
export PYTHONPATH=/home/a/aspuru/pascalf/codes/sterimol_db:$PYTHONPATH
export PATH="/home/a/aspuru/pascalf/codes/chemaxon/marvinsuite/bin:$PATH"
export PATH=$(pwd):$PATH

