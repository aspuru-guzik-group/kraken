import os,time,sys
import numpy as np
# import check_convergence_conformers as ccc
import yaml
from yaml import CLoader as Loader
import pandas as pd
import uuid
import matplotlib.pyplot as plt

import subprocess

import pyximport
pyximport.install()
import ConfPruneIdx as ConfPrune

import itertools
from rdkit import Chem,Geometry
from rdkit.Chem import rdmolfiles, AllChem, rdMolAlign, rdmolops


import openbabel
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("xyz", "sdf")

import PL_gaussian_input_200411 as PL_gaussian_input

from joblib import Parallel,delayed
import multiprocessing
nproc = 5#multiprocessing.cpu_count() - 4                                   

masses = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                 'OG' : 294}

rcov = { #*
"H": 0.32,"He": 0.46,"Li": 1.2,"Be": 0.94,"B": 0.77,"C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,"Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,"Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,"Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,"Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,"Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,"Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.08,"Ag": 1.15,"Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,"Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,"Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,"Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,"Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,"Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,"Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,"Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55 #*
} #*

def get_conmat(elements, coords):                                                                                  #*
    # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code       #*
    # elements is a list of strings, coords is a numpy array or nested list of shape N_atoms x 3                   #*
    if type(coords) == list:                                                                                       #*
        coords = np.asarray(coords)                                                                                #*
    natom = len(elements)                                                                                          #*
    #max_elem = 94                                                                                                 #*
    k1 = 16.0                                                                                                      #*
    k2 = 4.0/3.0                                                                                                   #*
    conmat = np.zeros((natom,natom))                                                                               #*
    for i in range(0,natom):                                                                                       #*
        if elements[i] not in rcov.keys():                                                                         #*
            continue                                                                                               #*
        for iat in range(0,natom):                                                                                 #*
            if elements[iat] not in rcov.keys():                                                                   #*
                continue                                                                                           #*
            if iat != i:                                                                                           #*
                dxyz = coords[iat]-coords[i]                                                                       #*
                r = np.linalg.norm(dxyz)                                                                           #*
                rco = rcov[elements[i]]+rcov[elements[iat]]                                                        #*
                rco = rco*k2                                                                                       #*
                rr=rco/r                                                                                           #*
                damp=1.0/(1.0+np.math.exp(-k1*(rr-1.0)))                                                           #*
                if damp > 0.85: #check if threshold is good enough for general purpose                             #*
                    conmat[i,iat],conmat[iat,i] = 1,1                                                              #*
    return(conmat)                                                                                                 #*
                                                                                                                   
def write_xyz(filename,conf,data_here,writefile=True, inverse=False):
    if inverse:
        geometry_string = "".join([f'{atom:>3} {-data_here["confdata"]["coords"][conf][ind][0]:15f} {-data_here["confdata"]["coords"][conf][ind][1]:15f} {-data_here["confdata"]["coords"][conf][ind][2]:15f}\n' for ind,atom in enumerate(data_here["confdata"]["elements"][0][:-1])])

    else:
        geometry_string = "".join([f'{atom:>3} {data_here["confdata"]["coords"][conf][ind][0]:15f} {data_here["confdata"]["coords"][conf][ind][1]:15f} {data_here["confdata"]["coords"][conf][ind][2]:15f}\n' for ind,atom in enumerate(data_here["confdata"]["elements"][0][:-1])])
    if writefile:
        with open(f"{filename}.xyz","w", newline='\n') as f:
            f.write(f'{len(data_here["confdata"]["coords"][conf])-1}\n\n')
            f.write(geometry_string)
    return(geometry_string)

def mirror_mol(mol0):
    """Create mirror image of the 3D structure in an RDKit molecule object (assumes that one structure/conformer is present in the object)."""
    # Iris Guo    
    mol1 = Chem.RWMol(mol0)
    conf1 = mol1.GetConformers()[0]    # assumption: 1 conformer per mol
    cart0 = np.array(conf1.GetPositions())
    cart1 = -cart0
    for i in range(mol1.GetNumAtoms()):
        conf1.SetAtomPosition(i,Geometry.Point3D(cart1[i][0],cart1[i][1],cart1[i][2]))
    mol = mol1.GetMol()
    rdmolops.AssignAtomChiralTagsFromStructure(mol)
    return(mol)

# selection schemes
def select_random(suffixes,energies, coords_all, elements_all, properties_all,Sel):
    conformers_to_use = {}

    for suffix in suffixes:
        n_conformers = len(energies[suffix])
        N_max=Sel.PerStructConfLimit
        n_conformers_energy = len([i for i in energies[suffix] if i < Sel.energycutoff])
        N = min(N_max, n_conformers_energy) # energycutoff can be set to something > 6 to reset original behavior
        conformers_to_use[suffix] = [0]
        if n_conformers_energy != 1:
            conformers_to_use[suffix] += sorted(np.random.choice(list(range(1,n_conformers_energy)), size=N-1, replace=False).tolist())
    return(conformers_to_use)

def select_RMSD(suffixes,energies, coords_all, elements_all, properties_all,Sel):
    rmsdconfs = {}

    for suffix in suffixes:
        # select from lower energy range
        num_conformers_le = len([i for i in energies[suffix] if i < Sel.energycutoff]) 

        PerStructConfLimit = max((Sel.PerStructConfLimit,int(np.log(num_conformers_le)**2)))

        if num_conformers_le > PerStructConfLimit:
            conformers = [[[elements_all[suffix][0][j],coords_all[suffix][i][j][0],coords_all[suffix][i][j][1],coords_all[suffix][i][j][2]] for j in range(len(elements_all[suffix][0]))] for i in range(num_conformers_le)] # convert geometry and elements to the format that ConfPrune expects
            pruned_le,pruned_indices_le, actualRMSDcutoff = ConfPrune.StrictRMSDPrune(conformers, elements_all[suffix][0], Sel.InitialRMSDcutoff, PerStructConfLimit) # pruned_le are the actual geometries, pruned_indices_le contains the indices of the selected conformers
        else:
            pruned_indices_le = [i for i in range(num_conformers_le)]
        
        if Sel.higher_energy:
            # select from higher energy range
            num_conformers_he = len(energies[suffix])-num_conformers_le
            if num_conformers_he > Sel.PerStructConfLimit:
                conformers = [[[elements_all[suffix][0][j],coords_all[suffix][i][j][0],coords_all[suffix][i][j][1],coords_all[suffix][i][j][2]] for j in range(len(elements_all[suffix][0]))] for i in range(num_conformers_le,num_conformers_he)]
                pruned_he,pruned_indices_tmp, actualRMSDcutoff = ConfPrune.StrictRMSDPrune(conformers, elements_all[suffix][0], Sel.InitialRMSDcutoff, Sel.PerStructConfLimit)
                pruned_indices_he = [i+num_conformers_le for i in pruned_indices_tmp]
            else:
                pruned_indices_he = [i for i in range(num_conformers_le,num_conformers_he)]
            rmsdconfs[suffix] = pruned_indices_le+pruned_indices_he

        else:
            rmsdconfs[suffix] = pruned_indices_le
    return(rmsdconfs)

def select_MinMax(suffixes, energies, coords_all, elements_all, properties_all, Sel):
    properties_df = pd.concat([pd.DataFrame(properties_all[suffix]) for suffix in suffixes], keys=suffixes) # Multilevel index with suffix at the first level and the conformer index at the second level

    nonproblematic_properties = ~properties_df[Sel.usepropsminmax].isna().any()
    props_argsort = np.argsort(properties_df[nonproblematic_properties[nonproblematic_properties==True].index],axis=0)

    use = [i for i in range(Sel.use_n)] + [i for i in range(-Sel.use_n,0)] # this allows to pick more than one conformer minimizing/maximizing each property
    absindices = sorted(set(props_argsort.iloc[use].values.reshape(-1))) # absolute indices of the min/max conformers in the Multilevel index
    setindices = [properties_df.index[i] for i in absindices] # indices of the min/max conformers within each ligand set. E.g. [("Ni",4),("noNi",0)]
    minmaxconfs = {i:[] for i in suffixes}
    [minmaxconfs[k].append(v) for k,v in setindices]

    return(minmaxconfs)

def select_all(suffixes,energies, coords_all, elements_all, properties_all,Sel):
    conformers_to_use = {}
    for suffix in suffixes:
        num_conformers_le = len([i for i in energies if i < Sel.energycutoff]) 
        conformers_to_use[suffix] = [i for i in range(num_conformers_le)]
    return(conformers_to_use)

class SelectionSettings:
# define selection schemes to use and which conformer set to apply that to
    selection_schemes = {
        select_RMSD:["noNi"],
        select_MinMax:["noNi","Ni"]
        }
    #options:
    # select_random   # random selection within energycutoff
    # select_RMSD     # RMSD clustering
    # select_MinMax   # conformers min/maxing properties defined in usepropsminmax
    # select_all      # all conformers within energycutoff

    # for min/max
    use_n = 1  # the lowest/highest how many conformers to use
    usepropsminmax = ["B1","B5","lval","far_vbur","far_vtot","max_delta_qvbur","max_delta_qvtot","near_vbur","near_vtot","ovbur_max","ovbur_min","ovtot_max","ovtot_min","pyr_val","qvbur_max","qvbur_min","qvtot_max","qvtot_min","vbur"]

    # for RMSD/random/all
    PerStructConfLimit = 20  # up to how many conformers to select
    InitialRMSDcutoff = 0.5  # RMSD criterion to start the selection with. 0.5 or 1.0 are reasonable values. This is increased in 0.2 steps until the desired number of conformers is left
    energycutoff = 3   # select from the relative energy range up to this value in kcal/mol
    higher_energy = False # if False, conformers are selected with less than energycutoff relative energy. If True, the same number of conformers 

def conformer_selection(suffixes,energies, coords_all, elements_all, properties_all, Sel):
    conformers_to_use = {i: [] for i in suffixes}
    
    for selection_scheme,sel_suffixes in Sel.selection_schemes.items():
        todo_suffixes = list(set(suffixes)&set(sel_suffixes)) # this allows a set of conformers to be missing
        if len(todo_suffixes) == 0:
            todo_suffixes = suffixes
        newconfs = selection_scheme(todo_suffixes,energies, coords_all, elements_all, properties_all,Sel)
        for k,v in newconfs.items():
            conformers_to_use[k] += v
        # print(conformers_to_use)
    conformers_to_use = {k:sorted(set(v)) for k,v in conformers_to_use.items()}    # remove duplicates 

    return(conformers_to_use)

def delete_element_from_rdkitmol(mol,element_to_delete):
    """Delete all instances of an element in an RDKit molecule object. Arguments: an RDKit molecule object and a string of the element to be removed"""
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]    
    atoms_delete = [i for i in range(len(elements)) if elements[i] == element_to_delete]
    atoms_delete.sort(reverse=True) # delete from the end to avoid index problems
    e_mol = Chem.EditableMol(mol)
    for atom in atoms_delete:
        e_mol.RemoveAtom(atom)
    new_mol = e_mol.GetMol()
    return(new_mol)

def delete_haloalkane_halides(mol):
    """Remove halides in perhaloalkyl moieties. Match CX2 where both X are the same halide and there is no H at the same carbon, and delete both X from the molecule object."""
    halides = ["F","Cl","Br","I"]
    matches = ()
    for hal in halides:  # this is to avoid matching mixed halides
        matches += mol.GetSubstructMatches(Chem.MolFromSmarts(f"{hal}[CH0]({hal})")) # matches CX3 and --CX2-- . In --CF2Cl, this would match CF2 only, keeping Cl
    if len(matches) == 0:
        return(mol)
    match_atoms = set([i for sub in matches for i in sub]) # this still includes the carbon atoms
    del_halides = [i for i in match_atoms if mol.GetAtomWithIdx(i).GetSymbol() in halides]
    del_halides.sort(reverse=True) # delete from the end to avoid index problems
    e_mol = Chem.EditableMol(mol)
    for atom in del_halides:
        e_mol.RemoveAtom(atom)
    new_mol = e_mol.GetMol()
    return(new_mol)

#ignore (use single job)
def write_submit_gnuparallel(molname, num_nodes, num_todo, jobs_per_node, maindirectory, allocation):
    submit_template="""#!/bin/bash
{}
#SBATCH --nodes={}
#SBATCH --ntasks=40
#SBATCH --time=24:00:00
##SBATCH -o 00000003_SPhos_06_Ni.log
##SBATCH --mail-type=ALL
##SBATCH --mail-user=gabriel.gomes@utoronto.ca
##SBATCH --job-name testjob
##SBATCH --output=testjob_%j.out

#cat $SLURM_JOB_NODELIST
#cd $SLURM_SUBMIT_DIR

#module load gcc/7.3.0
#module load python/3.6.4-anaconda5.1.0

module load intel/2017.7 intelmpi/2017.7 
module load java/1.8.0_162
module load gnu-parallel

export PATH="/home/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"

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

#g16
export g16root="/project/a/aspuru/opt/gaussian"
gr=$g16root
export GAUSS_EXEDIR="$gr/g16C01/bsd:$gr/g16C01"
export GAUSS_LEXEDIR="$gr/g16C01/linda-exe"
export GAUSS_ARCHDIR="$gr/g16C01/arch"
export GAUSS_BSDDIR="$gr/g16C01/bsd"
export LD_LIBRARY_PATH="$GAUSS_EXEDIR:$LD_LIBRARY_PATH"
export PATH="$PATH:$gr/gauopen:$GAUSS_EXEDIR"
GAUSS_SCRDIR=$SCRATCH

HOSTS=$(scontrol show hostnames $SLURM_NODELIST | tr \'\\n\' ,)
parallel --env OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH,PYTHONPATH,GAUSS_EXEDIR,GAUSS_LEXEDIR,GAUSS_ARCHDIR,GAUSS_BSDDIR,g16root,OMP_STACKSIZE,GAUSS_SCRDIR --joblog slurm-$SLURM_JOBID.log -j {} -S $HOSTS --wd $PWD "run.py {{}}" ::: {{00000..{}}}

python end.py


""".format(allocation, num_nodes, jobs_per_node, str(num_todo-1).zfill(5))
    outfile=open("{}/{}/submit_gnuparallel.sh".format(maindirectory, molname), "w")
    outfile.write(submit_template)
    outfile.close()



def write_submit_singlejob(molname, confname, maindirectory, allocation):
    dirname=os.getcwd()
    submit_template="""#!/bin/bash
{}
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=24:00:00
##SBATCH -o 00000003_SPhos_06_Ni.log
##SBATCH --mail-type=ALL
##SBATCH --mail-user=gabriel.gomes@utoronto.ca
##SBATCH --job-name testjob
##SBATCH --output=testjob_%j.out

#cat $SLURM_JOB_NODELIST
#cd $SLURM_SUBMIT_DIR

#module load gcc/7.3.0
#module load python/3.6.4-anaconda5.1.0

module load intel/2017.7 intelmpi/2017.7 
module load java/1.8.0_162

export PATH="/home/a/aspuru/pascalf/codes/anaconda3/bin:$PATH"

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

#g16
export g16root="/project/a/aspuru/opt/gaussian"
gr=$g16root
export GAUSS_EXEDIR="$gr/g16C01/bsd:$gr/g16C01"
export GAUSS_LEXEDIR="$gr/g16C01/linda-exe"
export GAUSS_ARCHDIR="$gr/g16C01/arch"
export GAUSS_BSDDIR="$gr/g16C01/bsd"
export LD_LIBRARY_PATH="$GAUSS_EXEDIR:$LD_LIBRARY_PATH"
export PATH="$PATH:$gr/gauopen:$GAUSS_EXEDIR"
GAUSS_SCRDIR=$SCRATCH

g16 <{}/{}/{}/{}/{}_singlejob.com>  {}/{}/{}/{}/{}.log
formchk {}/{}/{}/{}/{}.chk {}/{}/{}/{}/{}.fchk
vmin3.py {}.fchk


""".format(allocation, dirname, maindirectory, molname, confname, confname, dirname, maindirectory, molname, confname, confname, dirname, maindirectory, molname, confname, confname, dirname, maindirectory, molname, confname, confname, confname)
    outfile=open("{}/{}/{}/submit.sh".format(maindirectory, molname, confname), "w")
    outfile.write(submit_template)
    outfile.close()



def write_runpy(molname, maindirectory):
    submit_template="""#!/usr/bin/env python
import os
import sys
import os
import sys
sys.path.append("../../")
import PL_dft_library_201027
import pathlib as pl

idx = int(sys.argv[1])

confname = None
for lineidx, line in enumerate(open("todo.txt","r")):
    if lineidx==idx:
        confname = line.split()[0]
        break
if confname is None:
    print("ERROR: no confname found: %i"%(idx))
    exit()
if not os.path.exists(confname):
    print("ERROR: no confname directory found: %s"%(confname))
    exit()

startdir = os.getcwd()
os.chdir(confname)
actualdir = os.getcwd()
os.system("g16 <%s/%s.com>  %s/%s.log"%(actualdir, confname, actualdir, confname))
os.system("formchk %s/%s.chk %s/%s.fchk"%(actualdir, confname, actualdir, confname))
os.system("vmin4.py %s.fchk"%(confname))   ##changed from vmin3.py 5/7/21 by EP
os.chdir(startdir)


"""
    outfile=open("{}/{}/run.py".format(maindirectory, molname), "w")
    outfile.write(submit_template)
    outfile.close()
    os.system("chmod +x {}/{}/run.py".format(maindirectory, molname))



def write_endpy(molname, maindirectory):
    submit_template="""#!/usr/bin/env python
import os
import sys
import os
import sys
sys.path.append("../../")
import PL_dft_library_201027 ##changed from PL_dft_library 5/17/21 by EP
import pathlib as pl


os.system("rm */*.chk")
os.system("rm log_file_analysis")
os.system("for i in $(find */*.log);do echo $i >> log_file_analysis;grep \\\"Normal termination\\\" $i | wc -l >> log_file_analysis;done;")

startdir = os.getcwd()
ligand_name = startdir.split("/")[-1]

os.chdir("../")
cwd = pl.Path.cwd()
confnames = PL_dft_library.main_split_logs(cwd, ligand_name)
os.chdir(startdir)

confnames_all = [i.name for i in (cwd/ligand_name).iterdir() if i.is_dir() and ligand_name in i.name]
confnames = []
for n in confnames_all:
    err=False
    for x in os.listdir("../ERR"):
        if n in x:
            err=True
            break
    if not err:
        confnames.append(n)
confnames = sorted(confnames)

cwd = pl.Path.cwd()
PL_dft_library.read_ligand(cwd, ligand_name, confnames)

if not os.path.exists("../../dft_results"):
    os.makedirs("../../dft_results")

if os.path.exists("%s_confdata.yml"%(ligand_name)):
    os.system("cp %s_confdata.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_data.yml"%(ligand_name)):
    os.system("cp %s_data.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_relative_energies.csv"%(ligand_name)):
    os.system("cp %s_relative_energies.csv ../../dft_results/."%(ligand_name))

os.chdir("../")
os.system("zip -rq %s %s"%(ligand_name, ligand_name))
if os.path.exists("%s.zip"%(ligand_name)):
    os.system("rm -rf %s"%(ligand_name))


"""
    outfile=open("{}/{}/end.py".format(maindirectory, molname), "w")
    outfile.write(submit_template)
    outfile.close()
    os.system("chmod +x {}/{}/end.py".format(maindirectory, molname))



def get_rmsd_ext(fn1, fn2):
    #print(fn1)
    #print(fn2)
    randomstring=uuid.uuid4()
    out = "tempfiles/rmsd_%s.out"%(randomstring)
    #print(out)
    #os.system("calculate_rmsd --reorder --no-hydrogen %s %s > %s"%(fn1, fn2, out))
    #os.system("calculate_rmsd --reorder --print %s %s > %s"%(fn1, fn2, out))
    os.system("calculate_rmsd --reorder %s %s > %s"%(fn1, fn2, out))
    rmsd=float(open(out,"r").readlines()[0].split()[0])
    #exit()
    os.system("rm %s"%(out))
    return(rmsd)



def get_mass(elements):
    mass=0.0
    for el in elements:
        mass+=masses[el.upper()]
    return(mass)

def get_time(mass, params, deg):
    if deg==2:
        t = params[0]*mass**2.0+params[1]*mass+params[2]
    elif deg==3:
        t = params[0]*mass**3.0+params[1]*mass**2.0+params[2]*mass+params[3]
    return(t)



def conformer_selection_main(molname, alloc = "rrg-aspuru"):
    print('Hello, we are beginning conformer_selection_main!')
    warnings=""
    
    ###this section is for setting times on niagara - shouldn't need
    # read the reference times and fit to the curve
    #times=[]
    #for idx,line in enumerate(open("timing_new.csv","r")):
    #    if idx>0:
    #        times.append([float(line.split()[0].split(";")[4]),float(line.split()[0].split(";")[8])])
    #times = np.array(times)
    #deg=2
    #params = np.polyfit(times[:,0], times[:,1], 2)
    #
    #'''
    #ts=[]
    #ms = np.linspace(100,500,1000)
    #for m in ms:
    #    t = get_time(m, params, deg)
    #    ts.append(t)
    #plt.figure()
    #plt.scatter(times[:,0], times[:,1])
    #plt.plot(ms, ts, "k-")
    #plt.xlabel("moleceular weight")
    #plt.ylabel("cpu hours")
    #plt.savefig("times.png")
    #plt.close()
    #'''


    starttime_all = time.time()
    Sel = SelectionSettings()

    outfilename1="results_all_noNi/%s_combined.yml"%(molname)
    outfilename2="results_all_Ni/%s_combined.yml"%(molname)


    suffixes = []
    if os.path.exists(outfilename1):
        suffixes.append("noNi")
    if os.path.exists(outfilename2):
        suffixes.append("Ni")
    if len(suffixes) == 0:
        print("ERROR: no yml file found: \n  results_all_noNi/%s_combined.yml\n  results_all_Ni/%s_combined.yml"%(molname,molname))
        warnings+="ERROR: no yml file found: \n  results_all_noNi/%s_combined.yml\n  results_all_Ni/%s_combined.yml"%(molname,molname)
        return(False, warnings)




    maindirectory = "selected_conformers"
    if not os.path.exists(maindirectory):
        try:
            os.mkdir(maindirectory)
        except:
            pass

    if not os.path.exists("%s/%s"%(maindirectory, molname)):
        try:
            os.mkdir("%s/%s"%(maindirectory, molname))
        except:
            pass

    if not os.path.exists("tempfiles"):
        try:
            os.mkdir("tempfiles")
        except:
            pass

    if not os.path.exists("tempfiles/%s"%(molname)):
        try:
            os.mkdir("tempfiles/%s"%(molname))
        except:
            pass

    starttime = time.time()
    print("\n   ---   starting molecule {}".format(molname))
    conformers_to_use,number_of_conformers,data_here,energies,coords_all,elements_all,properties_all,sdffiles,conmats_all,conmat_check = {},{},{},{},{},{},{},{},{},{} #*
    for suffix in suffixes:
        print(f"{round((time.time()-starttime),2)} sec - Reading {suffix} results")
        outfilename="results_all_%s/%s_combined.yml"%(suffix, molname)
        outfile=open(outfilename,"r")
        data_here[suffix]=yaml.load(outfile,Loader = Loader)
        outfile.close()
        number_of_conformers[suffix] = data_here[suffix]["number_of_conformers"]
        energies[suffix] = data_here[suffix]["energies"]
        coords_all[suffix] = data_here[suffix]["confdata"]["coords"]
        elements_all[suffix] = data_here[suffix]["confdata"]["elements"]
        properties_all[suffix] = data_here[suffix]["confdata"]
    # check for consistent structures - inconsistency can arise from "reactions" during Crest conformational searches        #*
        conmats_all[suffix] = [get_conmat(elements_all[suffix][i][:-1],coords_all[suffix][i][:-1]) for i in range(number_of_conformers[suffix])]#*
        conmat_check[suffix] = np.zeros((len(conmats_all[suffix]),len(conmats_all[suffix])))                                 #*
        for i,j in zip(range(len(conmats_all[suffix])),range(len(conmats_all[suffix]))):                                     #*
            if np.shape(conmats_all[suffix][i]) != np.shape(conmats_all[suffix][j]): # different number of atoms: removing Ni-fragment failed #*
                conmat_check[suffix][i,j] = 1                                                                                #*
                conmat_check[suffix][j,i] = 1                                                                                #*
            elif np.abs(conmats_all[suffix][i]-conmats_all[suffix][j]).sum() != 0.0: # different connectivity matrix: bonding changes #*
                conmat_check[suffix][i,j] = 1                                                                                #*
                conmat_check[suffix][j,i] = 1                                                                                #*
        if np.sum(conmat_check[suffix]) != 0:                                                                                #*
            print("ERROR: %s %s has inconsistent structures, check manually."%(molname,suffix))                              #*
            warnings+="ERROR: %s %s has inconsistent structures, check manually.\n"%(molname,suffix)                         #*
            return(False, warnings)                                                                                          #*
                                                                                                                             #*
    if len(suffixes) == 2:                                                                                                   #*
        conmat_check_cross = np.zeros((len(conmats_all["noNi"]),len(conmats_all["Ni"])))                                     #*
        for i,j in zip(range(len(conmats_all["noNi"])),range(len(conmats_all["Ni"]))):                                       #*
            if np.shape(conmats_all["noNi"][i]) != np.shape(conmats_all["Ni"][j]):                                           #*
                conmat_check_cross[i,j] = 1                                                                                  #*
            elif np.abs(conmats_all["noNi"][i]-conmats_all["Ni"][j]).sum() != 0.0:                                           #*
                conmat_check_cross[i,j] = 1                                                                                  #*
        # if np.sum(conmat_check_cross) == np.shape(conmat_check_cross)[0]*np.shape(conmat_check_cross)[1]: # complete sets differ #*
        #     print("ERROR: %s conformer sets are not the same, check manually"%(molname))                                   #*
        #     warnings+="ERROR: %s conformer sets are not the same, check manually\n"%(molname)                              #*
        #     return(False, warnings)                                                                                        #*
        if np.sum(conmat_check_cross) != 0:                                                                                  #*
            print("ERROR: %s has inconsistent structures between the conformer sets, check manually."%(molname))             #*
            warnings+="ERROR: %s has inconsistent structures between the conformer sets, check manually.\n"%(molname)        #*
            #return(False, warnings)                                                                                          #*
     
    print(f"{round((time.time()-starttime),2)} sec - Starting conformer selection")
    
    # # Ligand filter
    ##optionally remove conformer sets where crest made a reaction
  

    conformers_to_use = conformer_selection(suffixes,energies, coords_all, elements_all,properties_all,Sel)

    print(f"{round((time.time()-starttime),2)} sec - Writing temporary structure files")
    if len(suffixes) > 1: # conformers selected from both conformer searches - remove possible duplicates 
        for suffix in suffixes:    
            sdffiles[suffix] = []
            for conf in conformers_to_use[suffix]:
                filename = "tempfiles/{}/{}_{}_temp_{}".format(molname, molname, suffix, str(conf).zfill(4))
                filename_inv = "tempfiles/{}/{}_{}_temp_{}_inv".format(molname, molname, suffix, str(conf).zfill(4))
                write_xyz(filename, conf, data_here[suffix])
                write_xyz(filename_inv, conf, data_here[suffix], inverse=True)
                mol = openbabel.OBMol()
                obConversion.ReadFile(mol, "{}.xyz".format(filename))   
                obConversion.WriteFile(mol, "{}.sdf".format(filename))
                sdffiles[suffix].append(filename)



        print(f"{round((time.time()-starttime),2)} sec - Creating molobjects")
        molobjects = {}
        molxyzfilenames = {}
        runrmsd = True #*
        for suffix in suffixes:
            molobjects[suffix] = [rdmolfiles.MolFromMolFile(f"{sdf}.sdf", removeHs=False, strictParsing=False) for sdf in sdffiles[suffix]]
            molobjects[suffix] = [i for i in molobjects[suffix] if i != None]                                                                                   #*
            if len(molobjects[suffix]) == 0:                                                                                                                    #*
                print("WARNING: %s %s failed generating RDKit mol objects. Continuing conformer selection without duplicate detection."%(molname,suffix))       #*
                warnings+="WARNING: %s %s failed generating RDKit mol objects. Continuing conformer selection without duplicate detection.\n"%(molname,suffix)  #*
                runrmsd = False                                                                                                                                 #*
                break                                                                                                                                           #*
            molxyzfilenames[suffix] = [f"{sdf}.xyz" for sdf in sdffiles[suffix]]
            [AllChem.MMFFOptimizeMolecule(m) for m in molobjects[suffix]] # FF optimization. Optional: makes potential duplicate detection more robust when comparing conformers from different origins
            molobjects[suffix] = [Chem.RemoveHs(mol) for mol in molobjects[suffix]] # Remove all H: also optional but speeds up RMSD calculation
            #if molobjects[suffix][0].HasSubstructMatch(Chem.MolFromSmarts("C(F)(F)C(F)(F)")):
            #if molobjects[suffix][0].HasSubstructMatch(Chem.MolFromSmarts("C(F)(F)(F)")):
            # if len(molobjects[suffix][0].GetSubstructMatch(Chem.MolFromSmarts("C(F)(F)(F)"))) > 2:
            molobjects[suffix] = [delete_haloalkane_halides(mol) for mol in molobjects[suffix]]
            molobjects[suffix+"_inv"] = [mirror_mol(mol) for mol in molobjects[suffix]] # create mirror images of each conformer
            molxyzfilenames[suffix+"_inv"] = [f"{sdf}_inv.xyz" for sdf in sdffiles[suffix]]
        

        ## optional: remove temporary .xyz and .sdf files 


        if runrmsd == True:                                                                                         #*
            print(f"{round((time.time()-starttime),2)} sec - RMSD Matrix")                                          #*
            rmsd_matrix = np.zeros(([len(molobjects[i]) for i in suffixes]))                                        #*
            #rmsd_matrix2 = np.zeros(([len(molobjects[i]) for i in suffixes]))                                      #*
            #counter = 1                                                                                            #*
            #ntotal = len(molobjects["noNi"]) * len(molobjects["Ni"])                                               #*
            def proc_rmsd_matrix(mo1,mo2,moi1,i,j,starttime):
                if j == 0:
                    print(f"  {int((time.time()-starttime))} currently at {i} {j}")
                rmsd1 = rdMolAlign.GetBestRMS(mo1, mo2)
                rmsd1_inv = rdMolAlign.GetBestRMS(moi1, mo2)
                return([(i,j),min(rmsd1, rmsd1_inv)])
                
            print(f'{len(molobjects["noNi"])}x{len(molobjects["Ni"])} conformers, {len(molobjects["noNi"])*len(molobjects["Ni"])*2} rmsd combinations')
            try:
                pool = Parallel(n_jobs=nproc,verbose=0)
                parall = pool(delayed(proc_rmsd_matrix)(molobjects["noNi"][i], molobjects["Ni"][j],molobjects["noNi_inv"][i],i,j,starttime) for i,j in itertools.product(range(len(molobjects["noNi"])),range(len(molobjects["Ni"]))))   
                         
                for results in parall:
                    rmsd_matrix[results[0]] = results[1]
            except RuntimeError:
                print("WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection."%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))) #*
                warnings+="WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection.\n"%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))  #*
                rmsd_matrix[i,j] = 100 # forces this pair to be kept, can be found in the xxx_rmsd_matrix.csv   #*
            #for i,j in itertools.product(range(len(molobjects["noNi"])),range(len(molobjects["Ni"]))):              #*
            #    try:                                                                                                #*
            #        rmsd1 = rdMolAlign.GetBestRMS(molobjects["noNi"][i], molobjects["Ni"][j])                       #*
            #        rmsd1_inv = rdMolAlign.GetBestRMS(molobjects["noNi_inv"][i], molobjects["Ni"][j])               #*
            #        rmsd_matrix[i,j] = min(rmsd1, rmsd1_inv)                                                        #*
            #    except RuntimeError:                                                                                #*
            #        print("WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection."%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))) #*
            #        warnings+="WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection.\n"%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))  #*
            #        rmsd_matrix[i,j] = 100 # forces this pair to be kept, can be found in the xxx_rmsd_matrix.csv   #*
            #        # return(False, warnings)                                                                       #*
                                                                                                                    #*
                #rmsd2 = get_rmsd_ext(molxyzfilenames["noNi"][i], molxyzfilenames["Ni"][j])                         #*
                #rmsd2_inv = get_rmsd_ext(molxyzfilenames["noNi_inv"][i], molxyzfilenames["Ni"][j])                 #*
                #rmsd_matrix2[i,j] = min(rmsd2, rmsd2_inv)                                                          #*
                #print("%i of %i done: %.3f %.3f  %.3f %.3f"%(counter, ntotal, rmsd1, rmsd2, rmsd1_inv, rmsd2_inv)) #*
                #counter+=1                                                                                         #*
                                                                                                                    #*
            print(f"{round((time.time()-starttime),2)} sec - Write")                                                #*
            ## optional: save rmsd_matrix                                                                           #*
            df = pd.DataFrame(rmsd_matrix, columns=conformers_to_use["Ni"], index=conformers_to_use["noNi"])        #*
            df.to_csv("{}/{}/{}_rmsdmatrix.csv".format(maindirectory, molname, molname),sep=";")                    #*
                                                                                                                    #*
            remove = set(np.where(rmsd_matrix < 0.2)[1])                                                            #*
            conformers_to_use["Ni"] = [i for i in conformers_to_use["Ni"] if i not in remove]                       #*

    # num_todo=0
    # for suffix in suffixes:
        # for conf in conformers_to_use[suffix]:
            # num_todo+=1

    # # estimate time
    # elements = data_here[suffixes[0]]["confdata"]["elements"][0][:-1]
    # mass = get_mass(elements)
    # print("%s has a mass of %f"%(molname, mass))
    # t = get_time(mass, params, deg) # in cpu hours
    # safety_factor=1.5
    # t *= safety_factor
    # print("%s has an estimated run time of %.2f CPU hours incl. a safety factor of %.2f"%(molname, t, safety_factor))
    # print("%s has an estimated run time of %.2f node hours incl. a safety factor of %.2f"%(molname, t/40.0, safety_factor))
    
    # t/=40.0 # in node hours
    # if t>24.0:
        # print("ERROR: %s will probably not finish within 24 hours"%(molname))
        # warnings+="ERROR: %s will probably not finish within 24 hours\n"%(molname)
        # return(False, warnings)
    # if t>12.0:
        # print("WARNING: %s will probably not finish within 12 hours, need %i cpus for %i conformers"%(molname, num_todo, num_todo))
    # print("%i gaussian calculation have to be done"%(num_todo))
    # '''
    # # minimum number of processors per job, defined by 40/num_todo
    # if num_todo in [1,2,4,5,8,10,20]:
        # min_procs1 = 40//num_todo
    # elif num_todo == 3:
        # min_procs1 = 40
    # elif num_todo == 6:
        # min_procs1 = 20
    # elif num_todo == 7:
        # min_procs1 = 10
    # else:
        # min_procs1 = 40//num_todo
    # if min_procs1 == 0:
        # min_procs1 = 1
    # print("minimum number of processors as defined by 40/num_confs is %i"%(min_procs1))
    # # minimum number of processors per job, defined by estimated upped bond of runtime
    # min_procs2 = int(round(40.0*t/24.0))
    # if min_procs2 == 0:
        # min_procs2 = 1
    # print("minimum number of processors as defined by maximum runtime is %i"%(min_procs2))


    # if num_todo==1:
        # min_procs = 40
    # elif num_todo==2:
        

    # # max of both
    # min_procs = max(min_procs1, min_procs2)
    # print("minimum number of processors by both criteria: %i"%(min_procs))
    # # round up numbers that don't make sense (only 1,2,4,5,10,20 and 40 make sense)
    # if min_procs in [1,2,4,5,10,20,40]:
        # pass
    # elif min_procs == 3:
        # min_procs = 4
    # elif min_procs in [6, 7, 8, 9]:
        # min_procs = 10
    # elif min_procs < 20:
        # min_procs = 20
    # elif min_procs < 40:
        # min_procs = 40
    # print("minimum number of processors after rounding up to divisors of 40: %i"%(min_procs))
    # '''


    num_processors = 40  ### make this a user variable
    todolist=open("{}/{}/todo.txt".format(maindirectory, molname), "w", newline='\n')
    for suffix in suffixes:
        with open("{}/{}/confselection_minmax_{}.txt".format(maindirectory, molname, suffix),"a", newline='\n') as f:
            f.write("{};{}\n".format(molname, ';'.join([str(i) for i in conformers_to_use[suffix]])))
        for conf in conformers_to_use[suffix]:
            confname = "{}_{}_{}".format(molname, suffix, str(conf).zfill(5))
            todolist.write("{}\n".format(confname))
            if not os.path.exists("{}/{}/{}".format(maindirectory, molname, confname)):
                try:
                    os.mkdir("{}/{}/{}".format(maindirectory, molname, confname))
                except:
                    pass
            geometry_string = write_xyz("./{}/{}/{}/{}".format(maindirectory, molname, confname, molname), conf, data_here[suffix], writefile=False) # if writefile is False, this only returns the geometry as a string as required by the next function to write the .com file
            PL_gaussian_input.write_coms("./{}/{}/{}/".format(maindirectory, molname, confname), confname, "", geometry_string, "all", num_processors)
            #PL_gaussian_input.write_coms("./{}/{}/{}/".format(maindirectory, molname, confname), confname, "_singlejob", geometry_string, "all", num_processors)
            
            
            ##do what is necessary here to write a submit script for your cluster
            
            #toronto
            #allocation = "#SBATCH --account=ctb-ccgem"
            #allocation = "#SBATCH --account=rrg-aspuru"
            #allocation = "#SBATCH --account=%s"%(alloc)  ###used for write submit job later on
            #write_submit_singlejob(molname, confname, maindirectory, allocation)
            
            #utah
            walltime = 24  ### make this a user-defined variable too
            
            sub_script_path = "/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/sub16_PL"
            ####### fix this here!!!
            sub_script_dir = f"/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/{maindirectory}/{molname}/{confname}"
           
            #print(sub_script_path, sub_script_dir)
          
            run_submit_script = subprocess.run(f"{sub_script_path} {sub_script_dir} {walltime}",stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="ascii",shell=True)


    todolist.close()





    # # get the number of nodes required
    # jobs_per_node = 40//num_processors
    # # estimated linear scaling based on time on 40 cores
    # num_done_in_24_per_node = int(round(24.0/t-0.5))*jobs_per_node
    # num_nodes_ideal = num_todo/num_done_in_24_per_node
    # num_nodes=int(round(num_nodes_ideal + 0.5))
    # if num_nodes==0:
        # num_nodes=1
    # #num_nodes = min(num_todo, 5)
    ##write_runpy(molname, maindirectory)   #commented out by EP 5/18/2021
    write_endpy(molname, maindirectory)

    #print("   ---   all done with %s. %i jobs todo on %i nodes, %.2f hours per Gaussian job. Total time: %.2f sec"%(molname, num_todo, num_nodes, t, round((time.time()-starttime_all),2)))   ##num_todo is not defined rn. fix or remove this
    print("  ---   yeehaw")
    return(True, warnings)


if __name__ == "__main__":
    molname = sys.argv[1]
    conformer_selection_main(molname)

