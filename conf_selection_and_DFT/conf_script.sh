#!/bin/bash
#SBATCH --partition=notchpeak
#SBATCH --account=sigman
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --job-name conformer-selection
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N


module load gcc/8.3.0


#export PATH="/scratch/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"
module load anaconda3/2019.03
conda activate peters_PL1


#export PYTHONPATH="/home/a/aspuru/pascalf/codes/morfeus:$PYTHONPATH"
##path2=`python3 -c'import path_defs_EP; path_defs_EP.path_defs_sh("conf_script_path2_pathEP")'`
#echo $path2
#export PYTHONPATH="/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org/pascalf_codes_morfeus:$PYTHONPATH"
## keep an eye out for this if morfeus doesn't work later


#export PYTHONPATH="/project/a/aspuru/passos/ligands_final_batch1_2020_02:$PYTHONPATH"
export PYTHONPATH="/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use:$PYTHONPATH"


ulimit -s unlimited

#python ../python_scripts/conformer_selection.py
python /uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/conformer_selection.py