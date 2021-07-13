from __future__ import print_function
from __future__ import absolute_import
import shutil
import uuid
import getpass
import socket
import os
import sys
import numpy as np
import scipy.spatial as scsp
import copy
import time
import subprocess
import shlex
import yaml


from morfeus import BuriedVolume
from morfeus import Pyramidalization
from morfeus import ConeAngle
from morfeus import Sterimol
from morfeus import SASA
from morfeus import Dispersion

import scipy.spatial as scsp
import scipy.linalg as scli
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi



kcal_to_eV=0.0433641153
kB=8.6173303e-5 #eV/K
T=298.15
kBT=kB*T
AToBohr=1.889725989



def get_bonds(coords, elements, force_bonds=False, forced_bonds=[]):

    # covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
    # values for metals decreased by 10% according to Robert Paton's Sterimol implementation
    rcov = {
    "H": 0.34,"He": 0.46,"Li": 1.2,"Be": 0.94,"B": 0.77,"C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,"Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,"Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,"Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,"Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,"Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,"Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.19,"Ag": 1.15,"Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,"Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,"Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,"Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,"Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,"Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,"Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,"Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55
    }

    # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    natom = len(coords)
    #max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    bonds = []
    for i in range(0,natom):
        if elements[i] not in rcov.keys():
            continue
        for iat in range(0,natom):
            if elements[iat] not in rcov.keys():
                continue
            if iat != i:
                dx = coords[iat][0] - coords[i][0]
                dy = coords[iat][1] - coords[i][1]
                dz = coords[iat][2] - coords[i][2]
                r = np.linalg.norm([dx,dy,dz])
                rco = rcov[elements[i]]+rcov[elements[iat]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0+np.math.exp(-k1*(rr-1.0)))
                if damp > 0.85: #check if threshold is good enough for general purpose
                    conmat[i,iat],conmat[iat,i] = 1,1
                    pair=[min(i,iat),max(i,iat)]
                    if pair not in bonds:

                        # add some empirical rules here:
                        is_bond=True
                        elements_bond = [elements[pair[0]], elements[pair[1]]]
                        if "Pd" in elements_bond:
                            if not ("As" in elements_bond or "Cl" in elements_bond or "P" in elements_bond):
                                is_bond=False
                        elif "Ni" in elements_bond:
                            if not ("C" in elements_bond or "P" in elements_bond):
                                is_bond=False
                        elif "As" in elements_bond:
                            if not ("Pd" in elements_bond or "F" in elements_bond):
                                is_bond=False
                        if is_bond:
                            bonds.append(pair)

    # remove bonds in certain cases
    bonds_to_remove = []
    # P has too many bonds incl one P-Cl bond which is probably to the spacer
    P_bond_indeces=[]
    P_bonds_elements=[]
    for bondidx, bond in enumerate(bonds):
        elements_bond=[elements[bond[0]],elements[bond[1]]]
        if "P" in elements_bond:
            P_bond_indeces.append(bondidx)
            P_bonds_elements.append(elements_bond)
    if len(P_bond_indeces)>4:
        print("WARNING: found a P with more than 4 bonds. try to remove one")
        if ["P","Cl"] in P_bonds_elements:
            bonds_to_remove.append(P_bond_indeces[P_bonds_elements.index(["P","Cl"])])
        elif ["Cl","O"] in P_bonds_elements:
            bonds_to_remove.append(P_bond_indeces[P_bonds_elements.index(["Cl","P"])])

    # Cl-Cl bonds
    for bondidx, bond in enumerate(bonds):
        elements_bond=[elements[bond[0]],elements[bond[1]]]
        if ["Cl", "Cl"] == elements_bond:
            bonds_to_remove.append(bondidx)

    bonds_new = []
    for bondidx, bond in enumerate(bonds):
        if bondidx not in bonds_to_remove:
            bonds_new.append(bond)
    bonds = bonds_new

    # very special case where the C atoms of Ni(CO)3 make additional bonds to lone pairs of N
    # get the indeces of the Ni(CO)3 C bonds
    c_atom_indeces = []
    for bondidx, bond in enumerate(bonds):
        elements_bond=[elements[bond[0]],elements[bond[1]]]
        if "Ni" == elements_bond[0] and "C" == elements_bond[1]:
            # check if this C has a bond to O
            for bondidx2, bond2 in enumerate(bonds):
                elements_bond2=[elements[bond2[0]],elements[bond2[1]]]
                if bond[1] in bond2 and "O" in elements_bond2:
                    c_atom_indeces.append(bond[1])
                    break
        elif "Ni" == elements_bond[1] and "C" == elements_bond[0]:
            for bondidx2, bond2 in enumerate(bonds):
                elements_bond2=[elements[bond2[0]],elements[bond2[1]]]
                if bond[0] in bond2 and "O" in elements_bond2:
                    c_atom_indeces.append(bond[0])
                    break

    if len(c_atom_indeces)>0:
        bonds_to_remove = []
        for c_atom_idx in c_atom_indeces:
            for bondidx, bond in enumerate(bonds):
                elements_bond=[elements[bond[0]],elements[bond[1]]]
                if c_atom_idx in bond and "N" in elements_bond:
                    bonds_to_remove.append(bondidx)
        bonds_new = []
        for bondidx, bond in enumerate(bonds):
            if bondidx not in bonds_to_remove:
                bonds_new.append(bond)
        bonds = bonds_new

    # add forced bonds
    if forced_bonds:
        for b in forced_bonds:
            b_to_add = [min(b),max(b)]
            if b_to_add not in bonds:
                print("WARNING: was forced to add a %s-%s bond that was not detected automatically."%(elements[b_to_add[0]],elements[b_to_add[1]]))
                bonds.append(b_to_add)


    # add bonds for atoms that are floating around
    indeces_used=[]
    for b in bonds:
        indeces_used.append(b[0])
        indeces_used.append(b[1])
    indeces_used=list(set(indeces_used))
    if len(indeces_used)<len(coords):
        for i in range(len(coords)):
            if i not in indeces_used:
                e = elements[i]
                c = coords[i]
                distances = scsp.distance.cdist([c],coords)[0]
                next_atom_indeces = np.argsort(distances)[1:]
                for next_atom_idx in next_atom_indeces:
                    b_to_add = [min([i, next_atom_idx]),max([i, next_atom_idx])]
                    elements_bond=[elements[b_to_add[0]],elements[b_to_add[1]]]
                    if elements_bond not in [["Cl","H"],["H","Cl"],["Cl","F"],["F","Cl"],["F","H"],["H","F"],["Pd","F"],["F","Pd"],["H","H"],["F","F"],["Cl","Cl"]]:
                        print("WARNING: had to add a %s-%s bond that was not detected automatically."%(elements[b_to_add[0]],elements[b_to_add[1]]))
                        bonds.append(b_to_add)
                        break
                    else:
                        pass
    return(bonds)




'''
def get_bonds(coords, elements):
    bondmax=1.7
    bondmax_special1=2.2
    bondmax_special2=2.8
    moltree=scsp.KDTree(coords)
    bonds=[]
    for atomidx,atom in enumerate(coords):
        #print(elements[atomidx])
        if elements[atomidx].capitalize() in ["Al","S","Si","P"]:
            #print("this is a special atom")
            bondmax_here=bondmax_special1
        elif elements[atomidx].capitalize() in ["Ir","Cu","Au","Pd","As", "Ni", "Fe"]:
            #print("this is a special atom")
            bondmax_here=bondmax_special2
        else:
            #print("this is a normal atom")
            bondmax_here=bondmax
        neighbours=moltree.query_ball_point(atom,bondmax_here)
        for neighbour in neighbours:
            if neighbour != atomidx:
                pair=[min(neighbour, atomidx),max(neighbour, atomidx)]
                if not pair in bonds:
                    elements_bond=[elements[pair[0]],elements[pair[1]]]
                    dist=np.linalg.norm(coords[pair[0]]-coords[pair[1]])
                    if "Pd" in elements_bond:
                        if "P" in elements_bond or "Cl" in elements_bond or "As" in elements_bond:
                            bonds.append(pair)

                    elif "Ni" in elements_bond:
                        if "P" in elements_bond or "C" in elements_bond:
                            bonds.append(pair)

                    elif "Fe" in elements_bond:
                        if "H" not in elements_bond:
                            bonds.append(pair)

                    elif "As" in elements_bond:
                        if "Pd" in elements_bond or "F" in elements_bond:
                            bonds.append(pair)

                    elif "P" in elements_bond and "H" in elements_bond:
                        if dist<bondmax:
                            bonds.append(pair)

                    elif "P" in elements_bond and ("C" in elements_bond or "O" in elements_bond or "N" in elements_bond):
                        if dist<bondmax_special1:
                            bonds.append(pair)
                    else:
                        bonds.append(pair)
    return(bonds)
'''

def separate_at_bond(coords, elements, bonds, bondidx, smiles):

    start1=bonds[bondidx][0]
    start2=bonds[bondidx][1]
    dihedral_atoms=[]
    connections1_all=[]
    connections1_to_check=[]
    for bondidx2,bond in enumerate(bonds):
        if bondidx2!=bondidx:
            if start1 == bond[0]:
                connection_new=bond[1]
            elif start1 == bond[1]:
                connection_new=bond[0]
            else:
                continue
            connections1_all.append(connection_new)
            connections1_to_check.append(connection_new)
    if len(connections1_to_check)==0:
        exit("ERROR: no metal-P dihedral found for %s"%(smiles))
    else:
        dihedral_atoms.append(connections1_to_check[0])

    dihedral_atoms.append(start1)
    dihedral_atoms.append(start2)

    while len(connections1_to_check)>0:
        for connection in connections1_to_check:
            for bondidx2,bond in enumerate(bonds):
                if bondidx2!=bondidx:
                    if connection == bond[0]:
                        connection_new=bond[1]
                    elif connection == bond[1]:
                        connection_new=bond[0]
                    else:
                        continue
                    if connection_new not in connections1_all and connection_new not in connections1_to_check:
                        connections1_to_check.append(connection_new)
                        connections1_all.append(connection_new)
            connections1_to_check.remove(connection)

    connections2_all=[]
    connections2_to_check=[]
    for bondidx2,bond in enumerate(bonds):
        if bondidx2!=bondidx:
            if start2 == bond[0]:
                connection_new=bond[1]
            elif start2 == bond[1]:
                connection_new=bond[0]
            else:
                continue
            connections2_all.append(connection_new)
            connections2_to_check.append(connection_new)
    if len(connections2_to_check)==0:
        exit("ERROR: no metal-P dihedral found for %s"%(smiles))
    else:
        dihedral_atoms.append(connections2_to_check[0])
    
    while len(connections2_to_check)>0:
        for connection in connections2_to_check:
            for bondidx2,bond in enumerate(bonds):
                if bondidx2!=bondidx:
                    if connection == bond[0]:
                        connection_new=bond[1]
                    elif connection == bond[1]:
                        connection_new=bond[0]
                    else:
                        continue
                    if connection_new not in connections2_all and connection_new not in connections2_to_check:
                        connections2_to_check.append(connection_new)
                        connections2_all.append(connection_new)
            connections2_to_check.remove(connection)
    connections1_all=sorted(connections1_all)
    connections2_all=sorted(connections2_all)
    return(connections1_all, connections2_all)


def get_ligand_indeces(coords, elements, P_index, smiles, metal_char):

    bonds = get_bonds(coords, elements, force_bonds=True, forced_bonds=[[P_index, elements.index(metal_char)]])
    #indeces=[]
    #for b in bonds:
    #    indeces.append(b[0])
    #    indeces.append(b[1])
    #indeces=list(set(indeces))
    #print(len(indeces))
    #for i in range(len(indeces)):
    #    if i not in indeces:
    #        print(i, elements[i])
    #exit()
    #print(len(bonds))

    #print(elements)
    #print(P_index)
    #for bondidx, bond in enumerate(bonds):
    #    elements_bond=[elements[bond[0]],elements[bond[1]]]
    #    print(elements_bond)
    #exit()

    found=False
    for bondidx, bond in enumerate(bonds):
        elements_bond=[elements[bond[0]],elements[bond[1]]]
        if metal_char in elements_bond and "P" in elements_bond and P_index in bond:
            found=True
            break
    if found:
        indeces1, indeces2 = separate_at_bond(coords, elements, bonds, bondidx, smiles)
        #print("group 1:")
        #for idx in indeces1:
        #    element_bond = elements[idx]
        #    print(element_bond)
        #print("group 2:")
        #for idx in indeces2:
        #    element_bond = elements[idx]
        #    print(element_bond)
        
        if metal_char==elements_bond[0]:
            mask=indeces2
        else:
            mask=indeces1
        #print(len(mask))
        #print(len(indeces1) + len(indeces2), len(indeces1), len(indeces2))
        #exit()
        return(mask, True)
    else:
        print("ERROR: No %s P bond found! %s"%(metal_char, smiles))
        return(None, False)

def sanitize_smiles(smi):
    return mol2smi(smi2mol(smi, sanitize=True), isomericSmiles=False, canonical=True)



def run_crest(coords, elements, moldir, filename, settings, smiles):
   
    startdir=os.getcwd()
    if settings["use_scratch"]:
        os.chdir(moldir)
        oldcwd, scratch_directory = goToScratch()
        oldmoldir=moldir
        moldir=scratch_directory
    else:
        os.chdir(moldir)

    exportXYZ(coords, elements, filename)
    #exit()
    time1=time.time()

    done=False
    if os.path.exists("crest.log"):
        for line in open("crest.log","r"):
            if "CREST terminated normally." in line:
                done=True
                break
        for x in os.listdir("."):
            if x.startswith("OPTIM"):
                if os.path.isdir(x):
                    try:
                        shutil.rmtree(x)
                    except:
                        pass
    if done and not os.path.exists("crest_best.xyz"):
        done=False


    if not done:
        call_crest(filename, settings)
    else:
        print("   ---   found old crest run and read output")
        pass
    crest_done, coords_all, elements_all, boltzmann_data = get_crest_results(settings)

    if len(elements_all)==0:
        exit("ERROR: No conformers found for %s"%(smiles))

    if "P" not in elements_all[0]:
        exit("ERROR: No P found in the first conformer of %s"%(smiles))


    P_index=elements_all[0].index("P")
    settings["P_index"]=P_index

    xtb_done=True
    time2=time.time()
    time_crest=time2-time1
    coords_all_used=[]
    elements_all_used=[]
    boltzmann_data_used=[]
    conf_indeces_used=[]
    if crest_done:
        electronic_properties_conformers=[]
        #conf_idx=0
        #done2=True
        #done3=True
        #while done2 and done3 and conf_idx<len(coords_all):

        for conf_idx in range(len(coords_all)):

            moldir2="conf_%i"%(conf_idx)
            try_mkdir(moldir2)
            startdir2=os.getcwd()
            os.chdir(moldir2)
            filename2="conf_%i.xyz"%(conf_idx)
            skip_this_conformer = False
            print("   ---   Run xtb calculation of molecule %s, conformer %i out of %i"%(filename,conf_idx+1,len(coords_all)))
            if settings["add_Pd_Cl2_PH3"] or settings["add_Pd_Cl2"] or settings["add_Ni_CO_3"]:
                P_index = settings["P_index"]
                mask, done = get_ligand_indeces(np.array(coords_all[conf_idx]),elements_all[conf_idx], P_index, smiles, settings["metal_char"])
                if not done:
                    exit()
                if settings["add_Ni_CO_3"] and len(mask)!=len(coords_all[conf_idx])-7:
                    print("WARNING: expected a mask of length 7 but got %i. Skip this conformer."%(len(coords_all[conf_idx])-len(mask)))
                    skip_this_conformer=True
            else:
                mask=[]

            if not skip_this_conformer:
                done = False
                if os.path.exists("xtb.log") or os.path.exists("xtb_ipea/xtb_ipea.log"):
                    done1 = False
                    for line in open("xtb.log", "r"):
                        if "wall-time" in line:
                            done1 = True
                            break
                    done2 = False
                    for line in open("xtb_ipea/xtb_ipea.log", "r"):
                        if "wall-time" in line:
                            done2 = True
                            break
                    if done1 and done2:
                        done=True
                    else:
                        done = False
                        os.system("rm -rf *")

                if not done:
                    exportXYZ(coords_all[conf_idx],elements_all[conf_idx],filename2, mask=mask)
                    call_xtb(filename2, settings)
                xtb_done_here, muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, esp_profile, esp_points, occ_energies, virt_energies, nucleophilicity = get_results_conformer()
                dummy_position_done_here, dummy_positions = get_dummy_positions()
                #print(xtb_done_here,dummy_position_done_here)
                electronic_properties_conformers.append({"muls":muls,
                                                         "alphas":alphas,
                                                         "wils":wils,
                                                         "dip":dip,
                                                         "alpha":alpha,
                                                         "dummy_positions":dummy_positions,
                                                         "fukui":fukui,
                                                         "HOMO_LUMO_gap":HOMO_LUMO_gap,
                                                         "IP_delta_SCC":IP_delta_SCC,
                                                         "EA_delta_SCC":EA_delta_SCC,
                                                         "global_electrophilicity_index":global_electrophilicity_index,
                                                         "esp_profile":esp_profile,
                                                         "esp_points":esp_points,
                                                         "occ_energies":occ_energies,
                                                         "virt_energies":virt_energies,
                                                         "nucleophilicity":nucleophilicity
                                                         })
                os.chdir(startdir2)
                #conf_idx+=1
                #print("did conf %i, len of data: %i"%(conf_idx,len(electronic_properties_conformers)))
                coords_all_used.append(coords_all[conf_idx])
                elements_all_used.append(elements_all[conf_idx])
                boltzmann_data_used.append(boltzmann_data[conf_idx])
                conf_indeces_used.append(conf_idx)
                if not xtb_done_here or not dummy_position_done_here:
                    xtb_done=False
    else:
        electronic_properties_conformers=[]

    time3=time.time()
    time_xtb_sterimol=time3-time2

    if settings["use_scratch"]:
        comeBachFromScratch(oldcwd, scratch_directory,settings)
        moldir=oldmoldir
        os.chdir(oldcwd)
    #else:
    #    os.chdir(moldir)

    os.chdir(startdir)
    return(crest_done, xtb_done, coords_all_used, elements_all_used, boltzmann_data_used, conf_indeces_used, electronic_properties_conformers, [time_crest, time_xtb_sterimol])


def get_dummy_positions():
    dummy_positions=[]
    done=False
    if os.path.exists("lmocent.coord"):
        for line in open("lmocent.coord","r"):
            if len(line.split())==4 and "He" in line:
                dummy_positions.append([float(line.split()[0])/AToBohr,float(line.split()[1])/AToBohr,float(line.split()[2])/AToBohr])
            if "$end" in line and len(dummy_positions)>0:
                done=True
                break
    if done:
        return(True, dummy_positions)
    else:
        return(False, None)


def read_crest_log():
    read=False
    data=[]
    for line in open("crest.log","r"):
        if "T /K" in line:
            read=False
        if read:
            if len(line.split())>=7:
                energy=float(line.split()[1])
                weight=float(line.split()[4])
                degen=int(line.split()[6])
                if len(line.split())==8:
                    origin=line.split()[7]
                else:
                    origin=None
                data.append({"energy":energy,"weight":weight,"degen":degen,"origin":origin})

        if "Erel/kcal     Etot      weight/tot conformer  set degen    origin" in line:
            read=True
    return(data)


def read_xtb_log():
    read_mul=False
    read_wil=False
    read_dip=False
    muls=[]
    alphas=[]
    wils=[]
    dip=[0.0,0.0,0.0]
    alpha=None
    fukui=[]
    read_fukui=False
    HOMO_LUMO_gap=None
    occ_energies=[]
    virt_energies=[]
    read_orbital_energies=False
    for line in open("xtb.log","r"):
        if "convergence criteria cannot be satisfied within" in line:
            break
        if read_mul and len(line.split())==0:
            read_mul=False
        if read_fukui and len(line.split())==0:
            read_fukui=False
        if read_wil and len(line.split())==0:
            read_wil=False
        if read_orbital_energies and len(line.split())==0:
            read_orbital_energies=False
        if read_dip and "molecular quadrupole" in line:
            read_dip=False
        if read_fukui:
            if len(line.split())>4:
                fukui.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4])])
        if read_mul:
            if len(line.split())==7:
                muls.append(float(line.split()[4]))
                alphas.append(float(line.split()[6]))
        if read_wil:
            if len(line.split())>2:
                if "*" in line.split()[2]:
                    wils.append(0.0)
                else:
                    wils.append(float(line.split()[2]))
        if read_dip and "full:" in line and len(line.split())>4:
            dip=[float(line.split()[1]),float(line.split()[2]),float(line.split()[3])]


        if read_orbital_energies and "occ." in line:
            occ=[]
            for x in line.split()[2:]:
                occ.append(int(round(float(x))))

        if read_orbital_energies and "eps" in line:
            es=[]
            for x in line.split()[2:]:
                es.append(float(x))
            if len(es)==len(occ):
                for idx in range(len(es)):
                    if occ[idx]>0:
                        occ_energies.append(es[idx])
                    else:
                        virt_energies.append(es[idx])


        if "Mol. α(0) /au        :" in line and len(line.split())==5:
            alpha=float(line.split()[4])
        if "#   Z        covCN         q      C6AA      α(0)" in line:
            read_mul=True
        if "total WBO             WBO to atom ..." in line:
            read_wil=True
        if "molecular dipole:" in line:
            read_dip=True
        if "#       f(+)     f(-)     f(0)" in line:
            read_fukui=True
        if "H-L gap (eV)  :" in line:
            HOMO_LUMO_gap=float(line.split()[4])
        if "eigenvalues" in line:
            read_orbital_energies=True


    global_electrophilicity_index=None
    EA_delta_SCC=None
    IP_delta_SCC=None
    empirical_EA_shift=None
    empirical_IP_shift=None
    for line in open("xtb_ipea/xtb_ipea.log","r"):
        if "convergence criteria cannot be satisfied within" in line:
            break
        if "Global electrophilicity index (eV):" in line:
            global_electrophilicity_index=float(line.split()[4])

        if "empirical EA shift (eV):" in line:
            empirical_EA_shift=float(line.split()[4])
        if "delta SCC EA (eV):" in line:
            EA_delta_SCC=float(line.split()[4])

        if "empirical IP shift (eV):" in line:
            empirical_IP_shift=float(line.split()[4])
        if "delta SCC IP (eV):" in line:
            IP_delta_SCC=float(line.split()[4])

    esp_profile=[]
    for line in open("xtb_esp_profile.dat","r"):
        if len(line.split())==2:
            esp_profile.append([float(line.split()[0]), float(line.split()[1])])

    esp_points=[]
    for line in open("xtb_esp.dat","r"):
        if len(line.split())==4:
            esp_points.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])

    nucleophilicity=-IP_delta_SCC

    return(muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, esp_profile, esp_points, occ_energies, virt_energies, nucleophilicity)


def read_xtb_log1():

    if not os.path.exists("xtb.log"):
        return(None, None, None, None, None, None, None, None, None, None, None)

    read_mul=False
    read_wil=False
    read_dip=False
    muls=[]
    alphas=[]
    wils=[]
    dip=[0.0,0.0,0.0]
    alpha=None
    fukui=[]
    read_fukui=False
    HOMO_LUMO_gap=None
    occ_energies=[]
    virt_energies=[]
    occ_done=False
    read_orbital_energies=False
    for line in open("xtb.log","r"):
        if "convergence criteria cannot be satisfied within" in line:
            break
        if read_mul and len(line.split())==0:
            read_mul=False
        if read_fukui and len(line.split())==0:
            read_fukui=False
        if read_wil and len(line.split())==0:
            read_wil=False
        if read_orbital_energies and "HL-Gap" in line:
            read_orbital_energies=False
        if read_dip and "molecular quadrupole" in line:
            read_dip=False
        if read_fukui:
            if len(line.split())>4:
                fukui.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4])])
        if read_mul:
            if len(line.split())==7:
                muls.append(float(line.split()[4]))
                alphas.append(float(line.split()[6]))
        if read_wil:
            if len(line.split())>2:
                wils.append(float(line.split()[2]))
        if read_dip and "full:" in line and len(line.split())>4:
            dip=[float(line.split()[1]),float(line.split()[2]),float(line.split()[3])]




        if read_orbital_energies and "-----" not in line and "..." not in line and len(line.split())!=0 and "Occupation" not in line:
            #print(line)
            num_entries=len(line.split())
            if "(HOMO)" in line or "(LUMO)" in line:
                num_entries-=1
            if not occ_done:
                if num_entries==4:
                    occ_energies.append(float(line.split()[3]))
                elif num_entries==3:
                    print("WARNING: error in parsing orbital energies")
                    occ_energies=[]
                    virt_energies=[]
                    read_orbital_energies=False
            else:
                if num_entries==4:
                    if "0.0000" in line:
                        pass
                    else:
                        print("WARNING: unexpected number of columns in parsing virtual energies")
                        print(line)
                    virt_energies.append(float(line.split()[3]))
                elif num_entries==3:
                    virt_energies.append(float(line.split()[2]))
            if "(HOMO)" in line:
                occ_done=True

        if "Mol. α(0) /au        :" in line and len(line.split())==5:
            alpha=float(line.split()[4])
        if "#   Z        covCN         q      C6AA      α(0)" in line:
            read_mul=True
            muls=[]
            alphas=[]
        if "total WBO             WBO to atom ..." in line:
            read_wil=True
            wils=[]
        if "molecular dipole:" in line:
            read_dip=True
        if "#       f(+)     f(-)     f(0)" in line:
            read_fukui=True
            fukui=[]
        if ":: HOMO-LUMO gap" in line:
            HOMO_LUMO_gap=float(line.split()[3])
        if "Orbital Energies and Occupations" in line:
            read_orbital_energies=True
            occ_energies=[]
            virt_energies=[]
            occ_done=False

    if len(occ_energies)==0:
        occ_energies=None
    else:
        occ_energies=occ_energies[1:]
    if len(virt_energies)==0:
        virt_energies=None
    else:
        virt_energies=virt_energies[:-1]
    if len(fukui)==0:
        fukui=None
    if len(wils)==0:
        wils=None
    if len(wils)==0:
        wils=None
    if len(alphas)==0:
        alphas=None

    if not os.path.exists("xtb_esp_profile.dat"):
        esp_profile=None
    else:
        esp_profile=[]
        for line in open("xtb_esp_profile.dat","r"):
            if len(line.split())==2:
                esp_profile.append([float(line.split()[0]), float(line.split()[1])])
        if len(esp_profile)==0:
            esp_profile=None
    if not os.path.exists("xtb_esp.dat"):
        esp_points=None
    else:
        esp_points=[]
        for line in open("xtb_esp.dat","r"):
            if len(line.split())==4:
                esp_points.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        if len(esp_points)==0:
            esp_points=None

    #print(HOMO_LUMO_gap, occ_energies, virt_energies)
    return(muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies)


def read_xtb_log2():

    if not os.path.exists("xtb_ipea/xtb_ipea.log"):
        return(None, None, None, None)
    #if "conf_3" in os.getcwd():
    #    return(None, None, None, None)

    global_electrophilicity_index=None
    EA_delta_SCC=None
    IP_delta_SCC=None
    empirical_EA_shift=None
    empirical_IP_shift=None
    nucleophilicity=None
    for line in open("xtb_ipea/xtb_ipea.log","r"):
        if "convergence criteria cannot be satisfied within" in line:
            break
        if "Global electrophilicity index (eV):" in line:
            global_electrophilicity_index=float(line.split()[4])

        if "empirical EA shift (eV):" in line:
            empirical_EA_shift=float(line.split()[4])
        if "delta SCC EA (eV):" in line:
            EA_delta_SCC=float(line.split()[4])

        if "empirical IP shift (eV):" in line:
            empirical_IP_shift=float(line.split()[4])
        if "delta SCC IP (eV):" in line:
            IP_delta_SCC=float(line.split()[4])
            nucleophilicity=-IP_delta_SCC

    return(IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity)


def get_crest_results(settings):
    done=False
    if os.path.exists("crest.log"):
        for line in open("crest.log","r"):
            if "CREST terminated normally." in line:
                done=True
    else:
        if os.path.exists("OPTIM"):
            os.system("rm -r OPTIM")
        if os.path.exists("METADYN1"):
            os.system("rm -r METADYN*")
        if os.path.exists("NORMMD1"):
            os.system("rm -r NORMMD*")
        exit("ERROR: DID NOT FIND CREST RESULTS: %s"%(os.getcwd()))
        #return(False, [], [], [])
    if done:
        coords_all, elements_all = readXYZs("crest_conformers.xyz")
        data = read_crest_log()
        #if len(coords_all)>1000:
        #    coords_all = coords_all[:1000]
        #    elements_all = elements_all[:1000]
        #    data = data[:1000]
        return(True, coords_all, elements_all, data)
    else:
        if os.path.exists("OPTIM"):
            os.system("rm -r OPTIM")
        if os.path.exists("METADYN1"):
            os.system("rm -r METADYN*")
        exit("ERROR: CREST MIGHT NOT HAVE FINISHED PROPERLY: %s"%(os.getcwd()))
        #return(False, [], [], [])


def get_results_conformer():
    done1=False
    if os.path.exists("xtb.log"):
        #for line in open("xtb.log","r"):
        #    if "finished run on" in line:
        done1=True
    done2=False
    if os.path.exists("xtb_ipea/xtb_ipea.log"):
        #for line in open("xtb_ipea/xtb_ipea.log","r"):
        #    if "finished run on" in line:
        done2=True

    if not done1 and not done2:
        return(False, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)

    #if done1:
    muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies = read_xtb_log1()
        
    #if done2:
    IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity = read_xtb_log2()

    return(True, muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, esp_profile, esp_points, occ_energies, virt_energies, nucleophilicity)


def xtb_opt(coords, elements, smiles, charge=0, freeze=[]):
    rundir="xtb_tmpdir_%s"%(uuid.uuid4())
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    else:
        if len(os.listdir(rundir))>0:
            os.system("rm %s/*"%(rundir))

    startdir=os.getcwd()
    os.chdir(rundir)
    exportXYZ(coords, elements, "in.xyz")

    if len(freeze)>0:
        outfile=open("xcontrol","w")
        outfile.write("$fix\n")
        outfile.write(" atoms: ")
        for counter,i in enumerate(freeze):
            if (counter+1)<len(freeze):
                outfile.write("%i,"%(i+1))
            else:
                outfile.write("%i\n"%(i+1))
        #outfile.write("$gbsa\n solvent=toluene\n")
        outfile.close()
        add=" -I xcontrol "
    else:
        add=""

    if charge==0:
        os.system("xtb %s in.xyz --opt >> xtb.log"%(add))
    else:
        os.system("xtb %s in.xyz --opt --chrg %i >> xtb.log"%(add,charge))
    if not os.path.exists("xtbopt.xyz"):
        print("WARNING: xtb geometry optimization did not work %s"%(smiles))
        os.chdir(startdir)
        os.system("rm -r %s"%(rundir))
        return(coords, elements)
            
    coords_new, elements_new=readXYZ("xtbopt.xyz")
    os.chdir(startdir)
    os.system("rm -r %s"%(rundir))
    return(coords_new, elements_new)





def call_crest(filename, settings):

    os.environ["OMP_NUM_THREADS"]="%s"%(settings["OMP_NUM_THREADS"])
    os.environ["MKL_NUM_THREADS"]="%s"%(settings["MKL_NUM_THREADS"])
    command="crest %s --gbsa toluene -metac -nozs"%(filename)
    #command="crest %s --gbsa toluene -metac"%(filename)
    #command="crest %s -ethr %f -pthi %f -metac"%(filename, settings["max_E"], settings["max_p"])
    # crest -chrg %i is used for charges
    args = shlex.split(command)
    mystdout = open("crest.log","a")

    process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
    out, err = process.communicate()
    mystdout.close()
    time.sleep(5)
    if settings["reduce_output"]:
        for x in os.listdir("."):
            if x.startswith("M") or x.startswith("N") or x=="wbo" or x=="coord" or "_rotamers_" in x or ".tmp" in x or x.startswith(".") or x=="coord.original":
                if os.path.isdir(x):
                    try:
                        shutil.rmtree(x)
                    except:
                        pass
                elif os.path.isfile(x):
                    try:
                        os.remove(x)
                    except:
                        pass
    return()


def call_xtb(filename, settings):

    os.environ["OMP_NUM_THREADS"]="%s"%(settings["OMP_NUM_THREADS"])
    os.environ["MKL_NUM_THREADS"]="%s"%(settings["MKL_NUM_THREADS"])
    command="xtb --gbsa toluene --lmo --vfukui --esp %s"%(filename)
    # check if xcontrol works
    args = shlex.split(command)
    mystdout = open("xtb.log","a")
    process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
    out, err = process.communicate()
    mystdout.close()
    if settings["reduce_output"]:
        for x in os.listdir("."):
            if "coordprot" in x or x=="wbo" or x=="xtbrestart" or x=="xtbscreen.xyz":
                if os.path.isdir(x):
                    try:
                        shutil.rmtree(x)
                    except:
                        pass
                elif os.path.isfile(x):
                    try:
                        os.remove(x)
                    except:
                        pass

    try_mkdir("xtb_ipea")
    startdir=os.getcwd()
    os.chdir("xtb_ipea")
    os.system("mv %s/%s ."%(startdir,filename))
    command="xtb --gbsa toluene --vomega --vipea %s"%(filename)
    # check if xcontrol works
    args = shlex.split(command)
    mystdout = open("xtb_ipea.log","a")
    process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
    out, err = process.communicate()
    mystdout.close()
    if settings["reduce_output"]:
        for x in os.listdir("."):
            if "coordprot" in x or x=="wbo" or x=="xtbrestart" or x=="xtbscreen.xyz" or x=="conf_0.xyz":
                if os.path.isdir(x):
                    try:
                        shutil.rmtree(x)
                    except:
                        pass
                elif os.path.isfile(x):
                    try:
                        os.remove(x)
                    except:
                        pass
    os.chdir(startdir)
    return()



def readXYZ(filename):
    infile=open(filename,"r")
    coords=[]
    elements=[]
    lines=infile.readlines()
    if len(lines)<3:
        exit("ERROR: no coordinates found in %s/%s"%(os.getcwd(), filename))
    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    coords=np.array(coords)
    return coords,elements



def readXYZs(filename):
    infile=open(filename,"r")
    coords=[[]]
    elements=[[]]
    for line in infile.readlines():
        if len(line.split())==1 and len(coords[-1])!=0:
            coords.append([])
            elements.append([])
        elif len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    return coords,elements

def exportXYZ(coords,elements,filename, mask=[]):
    outfile=open(filename,"w")

    if len(mask)==0:
        outfile.write("%i\n\n"%(len(elements)))
        for atomidx,atom in enumerate(coords):
            outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))
    else:
        outfile.write("%i\n\n"%(len(mask)))
        for atomidx in mask:
            atom = coords[atomidx]
            outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()

def exportXYZs(coords,elements,filename):
    outfile=open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()



def try_mkdir(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass



def run_sterimol(coords, elements, dummy_positions, moldir, settings, smiles):
    


    if settings["add_Pd_Cl2_PH3"] or settings["add_Pd_Cl2"] or settings["add_Ni_CO_3"]:
        P_index = settings["P_index"]
        metal_char=settings["metal_char"]

        mask, done = get_ligand_indeces(np.array(coords), elements, P_index, smiles, settings["metal_char"])
        if not done:
            exit()
        for idx,e in enumerate(elements):
            if e==metal_char:
                break
        pd_idx_full_ligand=idx
        
        # extend the molecule
        coords_list=[]
        elements_list=[]
        coords_extended=[]
        elements_extended=[]
        for atomidx in mask:
            atom = coords[atomidx]
            elements_extended.append(elements[atomidx])
            coords_extended.append([atom[0],atom[1],atom[2]])
            coords_list.append([atom[0],atom[1],atom[2]])
            elements_list.append(elements[atomidx])
        #print(coords_extended)
        coords_extended.append(coords[pd_idx_full_ligand])
        elements_extended+=[metal_char]

        for idx,e in enumerate(elements_extended):
            if e=="P":
                break
        p_idx=idx
        for idx,e in enumerate(elements_extended):
            if e==metal_char:
                break
        pd_idx=idx

        dummy_idx=len(elements_extended)-1

        outfile=open("%s/sterimol_input.xyz"%(moldir),"w")
        outfile.write("%i\nindeces: %i and %i\n"%(len(coords_extended),dummy_idx,p_idx))
        for idx,atom in enumerate(coords_extended):
            outfile.write("%s %f %f %f\n"%(elements_extended[idx],atom[0],atom[1],atom[2]))
        outfile.close()
        selected_dummy_idx=-1
    else:
        # get P position
        for idx,e in enumerate(elements):
            if e=="P":
                break
        p_idx=idx
        #print("p-idx: %i"%(p_idx))


        # get the correct dummy atom
        dummy_distances_to_p=scsp.distance.cdist([coords[p_idx]],dummy_positions)[0]
        #print(dummy_distances_to_p)
        #print("number of dummy positions: %i"%(len(dummy_positions)))
        nearest_dummy_indeces=np.argsort(dummy_distances_to_p)[:4]
        atom_distances_to_p=scsp.distance.cdist([coords[p_idx]],coords)[0]
        neighbor_indeces=np.argsort(atom_distances_to_p)[1:4]
        neighbor_dummy_distances=scsp.distance.cdist(np.array(dummy_positions)[nearest_dummy_indeces],np.array(coords)[neighbor_indeces])
        #print(neighbor_dummy_distances)
        minimal_distances=np.min(neighbor_dummy_distances,axis=1)
        #print(minimal_distances)
        dummy_atom_with_largest_minimal_distance=np.argmax(minimal_distances)
        selected_dummy_idx=nearest_dummy_indeces[dummy_atom_with_largest_minimal_distance]
        #print(selected_dummy_idx)
        
        # get the direction from P to dummy
        dummy_direction=np.array(dummy_positions[selected_dummy_idx])-np.array(coords[p_idx])
        dummy_direction_norm=np.linalg.norm(dummy_direction)
        dummy_direction/=dummy_direction_norm

        # go from p into dummy direction
        dummy_position=np.array(coords[p_idx])+settings["dummy_distance"]*dummy_direction


        # extend the molecule
        coords_list=[]
        elements_list=[]
        coords_extended=[]
        for atomidx,atom in enumerate(coords):
            coords_extended.append([atom[0],atom[1],atom[2]])
            coords_list.append([atom[0],atom[1],atom[2]])
            elements_list.append(elements[atomidx])
        #print(coords_extended)
        coords_extended.append(dummy_position.tolist())
        elements_extended=elements+["H"]
        dummy_idx=len(elements_extended)-1

        outfile=open("%s/sterimol_input.xyz"%(moldir),"w")
        outfile.write("%i\nindeces: %i and %i\n"%(len(coords_extended),dummy_idx,p_idx))
        for idx,atom in enumerate(coords_extended):
            outfile.write("%s %f %f %f\n"%(elements_extended[idx],atom[0],atom[1],atom[2]))
        outfile.close()

    
    # call sterimol OLD
    #file_Params = calcSterimol("", "bondi", dummy_idx+1, p_idx+1, False, get_coords=True, coords=coords_extended, elements=elements_extended)
    #lval_old = float(file_Params.lval)
    #B1_old = float(file_Params.B1)
    #B5_old = float(file_Params.newB5)
    #print("  %.2f"% lval, "  %.2f"% B1, "  %.2f"% B5)

    #print(len(coords_extended))
    #print(len(elements_extended))
    #print(dummy_idx)
    #print(p_idx)
    #print(elements_extended)


    # call steriplus NEW
    #print("coords_extended before steriplus: %i"%(len(coords_extended)))
    try:
        if len(elements_extended)!=len(coords_extended):
            print("WARNING: ConeAngle calculation got coords and elements with different sizes!")
        cone_angle = ConeAngle(elements_extended, coords_extended, dummy_idx+1)
        cone_angle_val = cone_angle.cone_angle
    except:
        cone_angle_val = 0.0

    #cone_angle.plot_3D()
    #print("coords_extended after cone: %i"%(len(coords_extended)))

    try:
        if len(elements_list)!=len(coords_list):
            print("WARNING: SASA calculation got coords and elements with different sizes!")
        sasa = SASA(elements_list, coords_list)
        sasa_val = sasa.area
        sasa_val_P = sasa.atom_areas[p_idx+1]
        sasa_volume = sasa.volume
        sasa_volume_P = sasa.atom_volumes[p_idx+1]
    except:
        sasa_val = 0.0
        sasa_val_P = 0.0
        sasa_volume = 0.0
        sasa_volume_P = 0.0
    #print("coords_extended after sasa: %i"%(len(coords_extended)))

    try:
        if len(elements_extended)!=len(coords_extended):
            print("WARNING: Sterimol calculation got coords and elements with different sizes!")
        sterimol = Sterimol(elements_extended, coords_extended, dummy_idx+1, p_idx+1)
        lval = sterimol.L_value
        B1 = sterimol.B_1_value
        B5 = sterimol.B_5_value
    except:
        lval = 0.0
        B1 = 0.0
        B5 = 0.0
    #print("coords_extended after sterimol: %i"%(len(coords_extended)))


    try:
        if len(elements_extended)!=len(coords_extended):
            print("WARNING: BuriedVolume calculation got coords and elements with different sizes!")
        bv = BuriedVolume(elements_extended, coords_extended, dummy_idx+1, exclude_list=[dummy_idx+1])
        buried_volume = bv.buried_volume
    except:
        buried_volume = 0.0

    try:
        if len(elements_list)!=len(coords_list):
            print("WARNING: Dispersion calculation got coords and elements with different sizes!")
        disp = Dispersion(elements_list, coords_list)
        p_int = disp.p_int
        p_int_atoms = disp.atom_p_ints
        p_int_atom = p_int_atoms[p_idx+1]
        p_int_area = disp.area
        p_int_atom_areas = disp.atom_areas
        p_int_atom_area = p_int_atom_areas[p_idx+1]
        p_int_times_p_int_area = p_int*p_int_area
        p_int_atom_times_p_int_atom_area = p_int_atom*p_int_atom_area
    except:
        p_int = 0.0
        p_int_atom = 0.0
        p_int_area = 0.0
        p_int_atom_area = 0.0
        p_int_times_p_int_area = 0.0
        p_int_atom_times_p_int_atom_area = 0.0


    results={"lval": float(lval),
             "B1": float(B1),
             "B5": float(B5),
             "buried_volume": float(buried_volume),
             "sasa": float(sasa_val),
             "sasa_P": float(sasa_val_P),
             "sasa_volume": float(sasa_volume),
             "sasa_volume_P": float(sasa_volume_P),
             "cone_angle": float(cone_angle_val),
             "p_int": float(p_int),
             "p_int_atom": float(p_int_atom),
             "p_int_area": float(p_int_area),
             "p_int_atom_area": float(p_int_atom_area),
             "p_int_times_p_int_area": float(p_int_times_p_int_area),
             "p_int_atom_times_p_int_atom_area": float(p_int_atom_times_p_int_atom_area),
             "selected_dummy_idx": int(selected_dummy_idx),
             "coords_extended": coords_extended,
             "elements_extended": elements_extended,
             "dummy_idx": int(dummy_idx),
             "p_idx": int(p_idx)
             }
    return(results)


def run_morfeus(coords, elements, dummy_positions, moldir, settings, smiles):

    outfilename="%s/morfeus.yml"%(moldir)

    if os.path.exists(outfilename):
        infile=open(outfilename,"r")
        results = yaml.load(infile, Loader=yaml.FullLoader)
        infile.close()
        return(results)

    times={}
    time0=time.time()
    do_pyramid=True
    if settings["add_Pd_Cl2_PH3"] or settings["add_Pd_Cl2"] or settings["add_Ni_CO_3"]:
        P_index = settings["P_index"]
        metal_char=settings["metal_char"]

        mask, done = get_ligand_indeces(np.array(coords), elements, P_index, smiles, settings["metal_char"])
        if not done:
            exit()
        for idx,e in enumerate(elements):
            if e==metal_char:
                break
        pd_idx_full_ligand=idx

        # extend the molecule
        coords_list=[]
        elements_list=[]
        coords_extended=[]
        elements_extended=[]
        for atomidx in mask:
            atom = coords[atomidx]
            elements_extended.append(elements[atomidx])
            coords_extended.append([atom[0],atom[1],atom[2]])
            coords_list.append([atom[0],atom[1],atom[2]])
            elements_list.append(elements[atomidx])
        #print(coords_extended)
        coords_extended.append(coords[pd_idx_full_ligand])

        elements_extended+=[metal_char]

        for idx,e in enumerate(elements_extended):
            if e=="P":
                break
        p_idx=idx
        for idx,e in enumerate(elements_extended):
            if e==metal_char:
                break
        pd_idx=idx


        dummy_idx=len(elements_extended)-1

        outfile=open("%s/sterimol_input.xyz"%(moldir),"w")
        outfile.write("%i\nindeces: %i and %i\n"%(len(coords_extended),dummy_idx,p_idx))
        for idx,atom in enumerate(coords_extended):
            outfile.write("%s %f %f %f\n"%(elements_extended[idx],atom[0],atom[1],atom[2]))
        outfile.close()


        atom_distances_to_p = scsp.distance.cdist([coords_extended[p_idx]],coords_extended)[0]
        neighbor_indeces = [i for i in np.argsort(atom_distances_to_p)[1:4] if i != pd_idx]
        
        if len(neighbor_indeces)!=3:
            print("WARNING: found %i instead of 3 neighbor indeces for ligand with %s (%s %s)"%(len(neighbor_indeces), metal_char, os.getcwd(), moldir))
            do_pyramid=False

        selected_dummy_idx=-1
    else:
        # get P position
        for idx,e in enumerate(elements):
            if e=="P":
                break
        p_idx=idx
        #print("p-idx: %i"%(p_idx))

        # get the correct dummy atom
        dummy_distances_to_p=scsp.distance.cdist([coords[p_idx]],dummy_positions)[0]
        #print(dummy_distances_to_p)
        #print("number of dummy positions: %i"%(len(dummy_positions)))
        nearest_dummy_indeces=np.argsort(dummy_distances_to_p)[:4]
        atom_distances_to_p=scsp.distance.cdist([coords[p_idx]],coords)[0]
        neighbor_indeces=np.argsort(atom_distances_to_p)[1:4]
        neighbor_dummy_distances=scsp.distance.cdist(np.array(dummy_positions)[nearest_dummy_indeces],np.array(coords)[neighbor_indeces])
        #print(neighbor_dummy_distances)
        minimal_distances=np.min(neighbor_dummy_distances,axis=1)
        #print(minimal_distances)
        dummy_atom_with_largest_minimal_distance=np.argmax(minimal_distances)
        selected_dummy_idx=nearest_dummy_indeces[dummy_atom_with_largest_minimal_distance]
        #print(selected_dummy_idx)
        
        # get the direction from P to dummy
        dummy_direction=np.array(dummy_positions[selected_dummy_idx])-np.array(coords[p_idx])
        dummy_direction_norm=np.linalg.norm(dummy_direction)
        dummy_direction/=dummy_direction_norm

        # go from p into dummy direction
        dummy_position=np.array(coords[p_idx])+settings["dummy_distance"]*dummy_direction

        # extend the molecule
        coords_list=[]
        elements_list=[]
        coords_extended=[]
        for atomidx,atom in enumerate(coords):
            coords_extended.append([atom[0],atom[1],atom[2]])
            coords_list.append([atom[0],atom[1],atom[2]])
            elements_list.append(elements[atomidx])
        #print(coords_extended)
        coords_extended.append(dummy_position.tolist())
        elements_extended=elements+["H"]
        dummy_idx=len(elements_extended)-1

        outfile=open("%s/sterimol_input.xyz"%(moldir),"w")
        outfile.write("%i\nindeces: %i and %i\n"%(len(coords_extended),dummy_idx,p_idx))
        for idx,atom in enumerate(coords_extended):
            outfile.write("%s %f %f %f\n"%(elements_extended[idx],atom[0],atom[1],atom[2]))
        outfile.close()

    time1=time.time()
    times["preparation"]=time1-time0

    # start morfeus stuff
    try:
        if len(elements_extended)!=len(coords_extended):
            print("WARNING: ConeAngle calculation got coords and elements with different sizes!")
        cone_angle = ConeAngle(elements_extended, coords_extended, dummy_idx+1)
        cone_angle_val = float(cone_angle.cone_angle)
    except:
        print("WARNING: morfeus cone angle failed")
        cone_angle_val = None

    time2=time.time()
    times["ConeAngle"]=time2-time1

    #cone_angle.plot_3D()
    #print("coords_extended after cone: %i"%(len(coords_extended)))

    try:
        if len(elements_list)!=len(coords_list):
            print("WARNING: SASA calculation got coords and elements with different sizes!")
        sasa = SASA(elements_list, coords_list)
        sasa_val = float(sasa.area)
        sasa_val_P = float(sasa.atom_areas[p_idx+1])
        sasa_volume = float(sasa.volume)
        sasa_volume_P = float(sasa.atom_volumes[p_idx+1])
    except:
        print("WARNING: morfeus sasa failed")
        sasa_val = None
        sasa_val_P = None
        sasa_volume = None
        sasa_volume_P = None
    #print("coords_extended after sasa: %i"%(len(coords_extended)))

    time3=time.time()
    times["SASA"]=time3-time2

    try:
        if len(elements_extended)!=len(coords_extended):
            print("WARNING: Sterimol calculation got coords and elements with different sizes!")
        sterimol = Sterimol(elements_extended, coords_extended, dummy_idx+1, p_idx+1)
        lval = float(sterimol.L_value)
        B1 = float(sterimol.B_1_value)
        B5 = float(sterimol.B_5_value)
    except:
        print("WARNING: morfeus sterimol failed")
        lval = None
        B1 = None
        B5 = None
    #print("coords_extended after sterimol: %i"%(len(coords_extended)))

    time4=time.time()
    times["Sterimol"]=time4-time3

    try:
        if len(elements_list)!=len(coords_list):
            print("WARNING: Dispersion calculation got coords and elements with different sizes!")
        disp = Dispersion(elements_list, np.array(coords_list))
        p_int = float(disp.p_int)
        p_int_atoms = disp.atom_p_int
        p_int_atom = float(p_int_atoms[p_idx+1])
        p_int_area = float(disp.area)
        p_int_atom_areas = disp.atom_areas
        p_int_atom_area = float(p_int_atom_areas[p_idx+1])
        p_int_times_p_int_area = float(p_int*p_int_area)
        p_int_atom_times_p_int_atom_area = float(p_int_atom*p_int_atom_area)
    except:
        print("WARNING: morfeus dispersion failed")
        p_int = None
        p_int_atom = None
        p_int_area = None
        p_int_atom_area = None
        p_int_times_p_int_area = None
        p_int_atom_times_p_int_atom_area = None

    time5=time.time()
    times["Dispersion"]=time5-time4

    # Pyramidalization - two equivalent measurments P and alpha
    if do_pyramid:
        try:
            pyr = Pyramidalization(elements = elements_extended, coordinates = coords_extended, atom_index = p_idx+1, excluded_atoms = [dummy_idx+1]) # remove Pd
            pyr_val = float(pyr.P)
            pyr_alpha = float(pyr.alpha)
        except:
            print("WARNING: morfeus Pyramidalization failed")
            pyr_val = None
            pyr_alpha = None
    else:
        pyr_val = None
        pyr_alpha = None

    time6=time.time()
    times["Pyramidalization"]=time6-time5
            
    #Buried volume - get quadrant volumes and distal volume 
    # iterate through P-substituents, aligning the quadrants paralell to each once (= xz_plane definition)
    # Metal/point of reference should be 2.28 A away from P
    # z_axis_atoms: P  
    # xz_plane_atoms: each of the substituents once
    # keep lowest and highest quadrant and octant volume across all three orientations of the coordinate system
    # keep highest difference of any neighboring quadrant volume
    # keep volume in each of the two hemispheres 

    try:
        qvbur_all = np.array([])
        qvdist_all = np.array([])
        qvtot_all = np.array([])
        max_delta_qvbur_all = []
        max_delta_qvtot_all = []
        ovbur_all = np.array([])
        ovtot_all = np.array([])

        for i in neighbor_indeces:  
            bv = BuriedVolume(elements_extended, coords_extended, dummy_idx+1, excluded_atoms=[dummy_idx+1], z_axis_atoms=[p_idx+1], xz_plane_atoms=[i+1], density=0.01) # dummy_idx+1 = 2
            bv.octant_analysis()
            bv.compute_distal_volume(method="buried_volume", octants=True)

            vbur = bv.buried_volume   # these are identical for each iteration
            #vbur = bv.percent_buried_volume   # these are identical for each iteration
            vdist = bv.distal_volume  #  
            vtot = vbur + vdist       #  

            qvbur = np.asarray(list(bv.quadrants["buried_volume"].values()))
            qvdist = np.asarray(list(bv.quadrants["distal_volume"].values()))
            qvtot = qvbur + qvdist
            
            qvbur_all = np.append(qvbur_all,qvbur)
            qvtot_all = np.append(qvtot_all,qvtot)

            max_delta_qvbur_all.append(max([abs(qvbur[i]-qvbur[i-1]) for i in range(4)]))
            max_delta_qvtot_all.append(max([abs(qvtot[i]-qvtot[i-1]) for i in range(4)]))

            ovbur = np.asarray(list(bv.octants["buried_volume"].values()))
            ovdist = np.asarray(list(bv.octants["distal_volume"].values()))
            ovtot = ovbur + ovdist

            ovbur_all = np.append(ovbur_all,ovbur)
            ovtot_all = np.append(ovtot_all,ovtot)

            near_vbur = ovbur[4:].sum()   # these are identical for each iteration
            far_vbur = ovbur[:4].sum()    # 
            near_vtot = ovtot[4:].sum()   # 
            far_vtot = ovtot[:4].sum()    # 
            
        qvbur_min = float(min(qvbur_all))
        qvbur_max = float(max(qvbur_all))
        qvtot_min = float(min(qvtot_all))
        qvtot_max = float(max(qvtot_all))

        max_delta_qvbur = float(max(max_delta_qvbur_all))
        max_delta_qvtot = float(max(max_delta_qvtot_all))

        ovbur_min = float(min(ovbur_all))
        ovbur_max = float(max(ovbur_all))
        ovtot_min = float(min(ovtot_all))
        ovtot_max = float(max(ovtot_all))

        # this is just a reminder to keep these properties
        vbur = float(vbur)
        vtot = float(vtot)
        near_vbur = float(near_vbur)
        far_vbur = float(far_vbur)
        near_vtot = float(near_vtot)
        far_vtot = float(far_vtot)


    except:
        print("WARNING: morfeus BuriedVolume failed")
        qvbur_min = None
        qvbur_max = None
        qvtot_min = None
        qvtot_max = None

        max_delta_qvbur = None
        max_delta_qvtot = None

        ovbur_min = None
        ovbur_max = None
        ovtot_min = None
        ovtot_max = None

        vbur = None
        vtot = None
        near_vbur = None
        far_vbur = None
        near_vtot = None
        far_vtot = None

    time7=time.time()
    times["BuriedVolume"]=time7-time6
    #print(times)


    results={"lval": lval,
             "B1": B1,
             "B5": B5,
             "sasa": sasa_val,
             "sasa_P": sasa_val_P,
             "sasa_volume": sasa_volume,
             "sasa_volume_P": sasa_volume_P,
             "cone_angle": cone_angle_val,
             "p_int": p_int,
             "p_int_atom": p_int_atom,
             "p_int_area": p_int_area,
             "p_int_atom_area": p_int_atom_area,
             "p_int_times_p_int_area": p_int_times_p_int_area,
             "p_int_atom_times_p_int_atom_area": p_int_atom_times_p_int_atom_area,
             "pyr_val": pyr_val,
             "pyr_alpha": pyr_alpha,
             "qvbur_min": qvbur_min,
             "qvbur_max": qvbur_max,
             "qvtot_min": qvtot_min,
             "qvtot_max": qvtot_max,
             "max_delta_qvbur": max_delta_qvbur,
             "max_delta_qvtot": max_delta_qvtot,
             "ovbur_min": ovbur_min,
             "ovbur_max": ovbur_max,
             "ovtot_min": ovtot_min,
             "ovtot_max": ovtot_max,
             "vbur": vbur,
             "vtot": vtot,
             "near_vbur": near_vbur,
             "far_vbur": far_vbur,
             "near_vtot": near_vtot,
             "far_vtot": far_vtot,
             "selected_dummy_idx": int(selected_dummy_idx),
             "coords_extended": coords_extended,
             "elements_extended": elements_extended,
             "dummy_idx": int(dummy_idx),
             "p_idx": int(p_idx)
             }

    outfilename="%s/morfeus.yml"%(moldir)
    outfile=open(outfilename, "w")
    outfile.write(yaml.dump(results, default_flow_style=False))
    outfile.close()


    return(results)




def copy_dir_contents_to_dir(in_directory, out_directory):
    try:
        #We copy all files in in_directory to out_directory without following directories recursively:
        for filename in os.listdir(in_directory):
            file_with_dir = "%s/%s" % (in_directory, filename)
            if os.path.isfile(file_with_dir):
                shutil.copy(file_with_dir, out_directory)
            elif os.path.isdir(file_with_dir):
                shutil.copytree(file_with_dir, "%s/%s"%(out_directory,filename))
    except Exception as exc:
        print("Moving files to from %s to %s has failed. Reraising Exception: %s." % (exc))
        raise


def copy_to_scratch(in_directory):
    try:
        #We generate the directory $SCRATCH/username/random-uuid
        SCRATCH_BASE = os.environ["SCRATCH"]
        username = getpass.getuser()
        randstring = uuid.uuid4()
        out_directory = "%s/%s/%s" % (SCRATCH_BASE, username, randstring)
        os.makedirs(out_directory)
        socketname = socket.gethostname()
        outfile = open("tmpdir.dat", "w")
        outfile.write("%s\n%s\n" % (socketname, out_directory))
        outfile.close()
        #and copy everything over:
        copy_dir_contents_to_dir(in_directory, out_directory)
        return in_directory, out_directory
    except IOError as exc:
        #In case of IOError or KeyError, we just return both times in-directory.
        print("Moving files to scratch has failed. Exception was IOError: %s. Turning off scratch handling." % (exc))
        return in_directory, in_directory
    except KeyError as exc:
        print("A KeyError occured, when querying the Scratch Directory. Check the environment settings. Exception was: %s. Turning off scratch handling." % ( exc ))
        return in_directory, in_directory
    except Exception as exc:
        #In case there was something unforseen, we reraise to bomb out of the application.
        print("An unexpected exception as occured of type %s. Exception was: %s. Reraising." % (type(exc), exc))
        raise

def goToScratch():
    oldcwd = os.getcwd()
    scratch_directory = ""
    #make scratch dir, enter scratch dir - save oldcwd
    oldcwd, scratch_directory = copy_to_scratch(oldcwd)
    os.chdir(scratch_directory)
    return [oldcwd, scratch_directory]


def comeBachFromScratch(oldcwd, scratch_directory,settings):
    if oldcwd != scratch_directory:
        #copy result back to oldcwd, change back, remove scratch
        copy_dir_contents_to_dir(scratch_directory, oldcwd)
        os.chdir(oldcwd)
        if settings["remove_scratch"]:
            try:
                shutil.rmtree(scratch_directory)
            except:
                pass
        os.system("rm tmpdir.dat")
    else:
        print("Warning, oldcwd ", oldcwd, "was equal to scratch_directory", scratch_directory, "review log for exceptions.")







def rotationMatrix(vector,angle):
    angle=angle/180.0*np.pi
    norm=(vector[0]**2.0+vector[1]**2.0+vector[2]**2.0)**0.5
    direction=vector/norm

    matrix=np.zeros((3,3))
    matrix[0][0]=direction[0]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[1][1]=direction[1]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[2][2]=direction[2]**2.0*(1.0-np.cos(angle))+np.cos(angle)

    matrix[0][1]=direction[0]*direction[1]*(1.0-np.cos(angle))-direction[2]*np.sin(angle)
    matrix[1][0]=direction[0]*direction[1]*(1.0-np.cos(angle))+direction[2]*np.sin(angle)

    matrix[0][2]=direction[0]*direction[2]*(1.0-np.cos(angle))+direction[1]*np.sin(angle)
    matrix[2][0]=direction[0]*direction[2]*(1.0-np.cos(angle))-direction[1]*np.sin(angle)

    matrix[1][2]=direction[1]*direction[2]*(1.0-np.cos(angle))-direction[0]*np.sin(angle)
    matrix[2][1]=direction[1]*direction[2]*(1.0-np.cos(angle))+direction[0]*np.sin(angle)

    return(matrix)


def overlap(coords1, coords_ref, idx1, idx2, elements):

    coords1_np=np.array(coords1)
    coords_ref_np=np.array(coords_ref)
    #print("overlap: coords1: %i, coords2: %i"%(len(coords1),len(coords_ref)))
    #print(idx1,idx2)

    # shift
    coords_shifted=coords1_np-coords1_np[idx2]+coords_ref_np[idx2]

    # rotate P-dummy-axis
    dir1=coords_shifted[idx1]-coords_shifted[idx2]
    dir1/=scli.norm(dir1)
    dir2=coords_ref_np[idx1]-coords_ref_np[idx2]
    dir2/=scli.norm(dir2)
    cross_dir1_dir2=np.cross(dir1,dir2)
    cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
    angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
    rotation=rotationMatrix(cross_dir1_dir2, angle)
    # shift to zero
    coords_shifted-=coords_shifted[idx2]
    coords_rotated=[]
    for atom in coords_shifted:
        coords_rotated.append(np.dot(rotation, atom).tolist())
    coords_rotated=np.array(coords_rotated)
    # shift back
    coords_rotated+=coords_ref_np[idx2]


    # rotate third axis
    axis2=coords_rotated[idx1]-coords_rotated[idx2]
    axis2/=scli.norm(axis2)
    RMSD_best=1e10
    angle2_best=0.0
    for angle2 in np.linspace(0.0,360.0,361):
        rotation2=rotationMatrix(axis2, angle2)
        # shift to zero
        coords_rotated-=coords_rotated[idx2]
        coords_rotated2=[]
        for atom in coords_rotated:
            coords_rotated2.append(np.dot(rotation2, atom))
        coords_rotated2=np.array(coords_rotated2)
        # shift back
        coords_rotated2+=coords_ref_np[idx2]
        RMSD=np.mean((coords_rotated2-coords_ref_np)**2.0)**0.5
        if RMSD<RMSD_best:
            RMSD_best=RMSD
            angle2_best=angle2
            #print("found better RMSD: %f"%(RMSD_best))

    rotation2=rotationMatrix(axis2, angle2_best)
    # shift to zero
    coords_rotated-=coords_rotated[idx2]
    coords_rotated_final=[]
    for atom in coords_rotated:
        coords_rotated_final.append(np.dot(rotation2, atom))
    coords_rotated_final=np.array(coords_rotated_final)
    # shift back
    coords_rotated_final+=coords_ref_np[idx2]
    #exportXYZs([coords_rotated_final,coords_ref_np],[elements+["H"],elements+["H"]],"test.xyz")
    return(coords_rotated_final.tolist())





def reduce_data(data_here):

    data_here["boltzmann_averaged_data"]={}
    data_here["min_data"]={}
    data_here["max_data"]={}
    data_here_esp_points={}

    confnames=[]
    counter=0
    for key in data_here.keys():
        if "conf_" in key:
            confnames.append("conf_%i"%(counter))
            counter+=1

    weights=[]
    energies=[]
    degeneracies=[]
    for confname in confnames:
        weights.append(data_here[confname]["boltzmann_data"]["weight"])
        degeneracies.append(data_here[confname]["boltzmann_data"]["degen"])
        energies.append(data_here[confname]["boltzmann_data"]["energy"]*kcal_to_eV)

    for confname in confnames:
        if not "elements" in data_here:
            data_here["elements"]=data_here[confname]["elements"]


    # own weight calculation for comparison
    # KEEP THIS CODE
    #print(weights)
    #print(np.sum(weights))
    #Z=np.sum(np.array(degeneracies)*np.exp(-np.array(energies)/kBT))
    #weights2=1.0/Z*np.array(degeneracies)*np.exp(-np.array(energies)/kBT)
    #print(weights2)

    # electronic_properties
    #######################
    keys_to_delete=[]
    for key in data_here["conf_0"]["electronic_properties"].keys():
        if key=="esp_points":
            data=[]
            min_q=-0.2
            max_q=0.2
            bins_q=np.linspace(min_q,max_q,50)
            binwidth=bins_q[1]-bins_q[0]
            for confname in confnames:
                try:
                    xyzq=np.array(data_here[confname]["electronic_properties"][key])
                    histdata=np.histogram(xyzq.T[3],bins=bins_q, density=True)[0]
                    data.append(histdata)
                except:
                    data.append(None)
                    print("ERROR in reduce data: esp_points")
                # shift it to the esp points dictionary
                if confname not in data_here_esp_points:
                    data_here_esp_points[confname]={}
                data_here_esp_points[confname][key]=data_here[confname]["electronic_properties"][key]

            try:
                data=np.array(data)
                data_averaged=np.average(data, weights=weights, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights, axis=0)**0.5

                data_here["boltzmann_averaged_data"]["esp_hist_bins"]=bins_q.tolist()
                data_here["boltzmann_averaged_data"]["esp_hist"]=data_averaged.tolist()
                data_here["boltzmann_averaged_data"]["esp_hist_std"]=data_std.tolist()
            except:
                data_here["boltzmann_averaged_data"]["esp_hist_bins"]=None
                data_here["boltzmann_averaged_data"]["esp_hist"]=None
                data_here["boltzmann_averaged_data"]["esp_hist_std"]=None
                print("ERROR in reduce data: esp_points")
            keys_to_delete.append(key)
        elif key=="esp_profile":
            # we calculate this with fixed bins, so we can remove the xtb data
            keys_to_delete.append(key)
        elif key=="dummy_positions":
            keys_to_delete.append(key)
        elif key=="dip":
            # averaging the dipole does not make too much sense because it can rotate completely from conf to conf
            # thus, we also average the norm of the dipole moment
            data=[]
            data_norm=[]
            for confname in confnames:
                data.append(data_here[confname]["electronic_properties"][key])
                data_norm.append(np.linalg.norm(data_here[confname]["electronic_properties"][key]))
            data=np.array(data)
            data_norm=np.array(data_norm)
            data_averaged=np.average(data,weights=weights,axis=0)
            data_averaged_norm=np.average(data_norm,weights=weights,axis=0)
            data_min_norm=np.min(data_norm,axis=0)
            data_max_norm=np.max(data_norm,axis=0)
            data_std = np.average((data-data_averaged)**2.0, weights=weights, axis=0)**0.5
            data_norm_std = np.average((data_norm-data_averaged_norm)**2.0, weights=weights, axis=0)**0.5
            data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
            data_here["boltzmann_averaged_data"][key+"_std"]=data_std.tolist()
            data_here["boltzmann_averaged_data"]["dip_norm"]=data_averaged_norm.tolist()
            data_here["boltzmann_averaged_data"]["dip_norm_std"]=data_norm_std.tolist()
            data_here["min_data"]["dip_norm"]=data_min_norm.tolist()
            data_here["max_data"]["dip_norm"]=data_max_norm.tolist()
        else:
            #print(key)
            data=[]
            weights_here=[]
            for confidx,confname in enumerate(confnames):
                x = data_here[confname]["electronic_properties"][key]
                if x is not None:
                    data.append(x)
                    #print(len(x))
                    weights_here.append(weights[confidx])
            #print(key)
            #print(data)
            if len(data)>0:
                data=np.array(data)
                weights_here=np.array(weights_here)
                #print(data)
                #print(data.shape)
                #print(weights_here)
                #print(weights_here.shape)
                data_averaged=np.average(data, weights=weights_here, axis=0)
                data_min=np.min(data, axis=0)
                data_max=np.max(data, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights_here, axis=0)**0.5
                #print(weights_here)
                #print(data_averaged.tolist(), data_min.tolist(), data_max.tolist())
                data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
                data_here["boltzmann_averaged_data"][key+"_std"]=data_std.tolist()
                data_here["min_data"][key]=data_min.tolist()
                data_here["max_data"][key]=data_max.tolist()
            else:
                data_here["boltzmann_averaged_data"][key]=None
                data_here["boltzmann_averaged_data"][key+"_std"]=None
                data_here["min_data"][key]=None
                data_here["max_data"][key]=None

    for key in keys_to_delete:
        for confname in confnames:
            del data_here[confname]["electronic_properties"][key]



    # morfeus_parameters
    #######################
    keys_to_delete=[]
    for key in data_here["conf_0"]["morfeus_parameters"].keys():
        if key=="elements_extended":
            elements_extended_list=[]
            for confname in confnames:
                elements_extended_list.append(data_here[confname]["morfeus_parameters"][key])
            pass
            #keys_to_delete.append(key)
        elif key=="selected_dummy_idx":
            keys_to_delete.append(key)
        elif key=="dummy_idx":
            pass
        elif key=="p_idx":
            pass
        elif key=="coords_extended":
            coords_extended_list=[]
            for confname in confnames:
                coords_extended_list.append(data_here[confname]["morfeus_parameters"][key])
            pass
        else:

            data=[]
            weights_here=[]
            for confidx,confname in enumerate(confnames):
                x = data_here[confname]["morfeus_parameters"][key]
                if x is not None:
                    data.append(x)
                    weights_here.append(weights[confidx])

            #print(key)
            #print(data)
            if len(data)>0:
                data=np.array(data)
                weights_here=np.array(weights_here)
                data_averaged=np.average(data, weights=weights_here, axis=0)
                data_min=np.min(data, axis=0)
                data_max=np.max(data, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights_here, axis=0)**0.5

                #print(weights_here)
                #print(data_averaged.tolist(), data_min.tolist(), data_max.tolist())
                data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
                data_here["boltzmann_averaged_data"][key+"_std"]=data_std.tolist()
                data_here["min_data"][key]=data_min.tolist()
                data_here["max_data"][key]=data_max.tolist()
            else:
                data_here["boltzmann_averaged_data"][key]=None
                data_here["boltzmann_averaged_data"][key+"_std"]=None
                data_here["min_data"][key]=None
                data_here["max_data"][key]=None

            #data=[]
            #for confname in confnames:
            #    data.append(data_here[confname]["morfeus_parameters"][key])
            #data=np.array(data)
            #data_averaged=np.average(data, weights=weights, axis=0)
            #data_min=np.min(data, axis=0)
            #data_max=np.max(data, axis=0)
            #data_std = np.average((data-data_averaged)**2.0, weights=weights, axis=0)**0.5

            #data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
            #data_here["boltzmann_averaged_data"][key+"_std"]=data_std.tolist()
            #data_here["min_data"][key]=data_min.tolist()
            #data_here["max_data"][key]=data_max.tolist()

            #keys_to_delete.append(key)
    for key in keys_to_delete:
        for confname in confnames:
            del data_here[confname]["morfeus_parameters"][key]


    # this code averages over the conformers
    # to do so, we first need to overlap the conformers as good as possible
    # for this, we shift and rotate the molecules in a way that their P and dummy atom position are the same
    # then we rotate around the P-dummy axis to minimize the RMSD
    coords_extended_list_rotated=[coords_extended_list[0]]
    for idx,coords_to_turn in enumerate(coords_extended_list[1:]):
        idx1=data_here["conf_0"]["morfeus_parameters"]["dummy_idx"]
        idx2=data_here["conf_0"]["morfeus_parameters"]["p_idx"]
        coords_turned = overlap(coords_to_turn, coords_extended_list_rotated[0], idx1, idx2, elements_extended_list[0])
        confname="conf_%i"%(idx+1)
        data_here[confname]["morfeus_parameters"]["coords_extended"]=coords_turned
        coords_extended_list_rotated.append(coords_turned)
    coords_extended_list_rotated=np.array(coords_extended_list_rotated)
    data_averaged=np.average(coords_extended_list_rotated, weights=weights, axis=0)
    data_std = np.average((coords_extended_list_rotated-data_averaged)**2.0, weights=weights, axis=0)**0.5
    data_here["boltzmann_averaged_data"]["coords_extended"]=data_averaged.tolist()
    data_here["boltzmann_averaged_data"]["coords_extended_std"]=data_std.tolist()



    # shift and delete some more data
    for confname in confnames:
        data_here[confname]["coords_extended"]=data_here[confname]["morfeus_parameters"]["coords_extended"]
        if "dummy_idx" not in data_here:
            data_here["dummy_idx"]=data_here[confname]["morfeus_parameters"]["dummy_idx"]
        if "p_idx" not in data_here:
            data_here["p_idx"]=data_here[confname]["morfeus_parameters"]["p_idx"]
        del data_here[confname]["coords"]
        del data_here[confname]["morfeus_parameters"]["coords_extended"]
        del data_here[confname]["morfeus_parameters"]["dummy_idx"]
        del data_here[confname]["morfeus_parameters"]["p_idx"]
        #del data_here[confname]["morfeus_parameters"]
        data_here[confname]["dip"]=data_here[confname]["electronic_properties"]["dip"]
        del data_here[confname]["electronic_properties"]["dip"]
        #del data_here[confname]["electronic_properties"]
        del data_here[confname]["elements"]

    data_here["number_of_conformers"]=len(confnames)
    data_here["boltzmann_weights"]=weights


    # move the conformer data to a separate dictionary
    data_confs={}
    for confname in confnames:
        data_confs[confname]=data_here[confname]
        del data_here[confname]


    return(data_here, data_confs, data_here_esp_points)




def get_weights(energies_here, degeneracies_here, selection=[]):
    T_kcal = 0.001987191686486*300.0
    if len(selection)==0:
        selection = np.array(list(range(len(degeneracies_here))))
    weights_own = np.array(degeneracies_here)[selection]*np.exp(-np.array(energies_here)[selection]/T_kcal)
    weights_own /= np.sum(weights_own)
    return(weights_own)



def combine_csvs(molname, resultsdir, data_here, data_here_confs):



    datagroups=["lval","B1","B5","sasa","sasa_P","sasa_volume","cone_angle",
                "global_electrophilicity_index","dip_norm","alpha","EA_delta_SCC",
                "HOMO_LUMO_gap","IP_delta_SCC","nucleophilicity", "cone_angle", "p_int", "p_int_atom", "p_int_area", "pyr_val", "pyr_alpha", "qvbur_min", "qvbur_max", "qvtot_min", "qvtot_max", "max_delta_qvbur", "max_delta_qvtot", "ovbur_min", "ovbur_max", "ovtot_min", "ovtot_max", "vbur", "vtot", "near_vbur", "far_vbur", "near_vtot", "far_vtot"]
    datagroups_vec=["muls","wils","fukui", "alphas"]

    ligand_data={}


    # read the boltzmann averages results files to get information about each ligand
    #outfilename="%s/%s.yml"%(resultsdir, molname)
    #print("   ---   read molecule %s"%(outfilename))
    #outfile=open(outfilename, "r")
    #data_here=yaml.load(outfile, Loader=yaml.FullLoader)
    #outfile.close()

    ligand_data[molname]={}
    ligand_data[molname]["number_of_atoms"] = len(data_here["elements"])
    ligand_data[molname]["num_rotatable_bonds"] = data_here["num_rotatable_bonds"]
    ligand_data[molname]["number_of_conformers"] = data_here["number_of_conformers"]
    ligand_data[molname]["smiles"] = data_here["smiles"]
    ligand_data[molname]["boltzmann_weights"] = data_here["boltzmann_weights"]
    for key in datagroups:
        ligand_data[molname][key+"_boltzmann"] = data_here["boltzmann_averaged_data"][key]
        ligand_data[molname][key+"_max"] = data_here["max_data"][key]
        ligand_data[molname][key+"_min"] = data_here["min_data"][key]
    p_idx=data_here["p_idx"]
    ligand_data[molname]["p_idx"] = p_idx
    for key in datagroups_vec:
        ligand_data[molname][key] = data_here["boltzmann_averaged_data"][key][p_idx]


    # read the conformer results files to get more information about each single conformer
    #outfilename_confs="%s/%s_confs.yml"%(resultsdir, molname)
    #outfile=open(outfilename_confs,"r")
    #data_here_confs=yaml.load(outfile, Loader=yaml.FullLoader)
    #outfile.close()

    n_conformers = ligand_data[molname]["number_of_conformers"]
    energies_here=[]
    degeneracies_here=[]
    weights_here=[]
    for c_idx in range(0,n_conformers):
        energies_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["energy"])
        degeneracies_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["degen"])
        weights_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["weight"])
    ligand_data[molname]["degeneracies"] = degeneracies_here
    ligand_data[molname]["energies"] = energies_here




    weights_own = get_weights(energies_here, degeneracies_here)

    # draw N random conformers (including lowest)
    N_max=10
    N = min(N_max, n_conformers)
    conformers_to_use = np.array([0] + sorted(np.random.choice(list(range(1,n_conformers)), size=N-1, replace=False).tolist()))
    weights_N = get_weights(energies_here, degeneracies_here, selection=conformers_to_use)



    coords_all = []
    elements_all = []
    for c_idx in range(0,n_conformers):
        #print(data_here_confs["conf_%i"%(c_idx)].keys())
        x = data_here_confs["conf_%i"%(c_idx)]["coords_extended"]
        e = data_here_confs["conf_%i"%(c_idx)]["morfeus_parameters"]["elements_extended"]
        coords_all.append(x)
        elements_all.append(e)
        #exportXYZ(x,e,"structures/single_files/%s_conformer_%i.xyz"%(molname, c_idx))
    coords_all = np.array(coords_all)
    #exportXYZs(coords_all,elements_all,"structures/%s_all_conformers.xyz"%(molname))

    ligand_data[molname]["confdata"]={}
    ligand_data[molname]["confdata"]["coords"] = coords_all.tolist()
    ligand_data[molname]["confdata"]["elements"] = elements_all


    electronic_properties = ['EA_delta_SCC', 'HOMO_LUMO_gap', 'IP_delta_SCC', 'alpha', 'alphas', 'global_electrophilicity_index', 'muls', 'nucleophilicity', 'wils']
    morfeus_parameters = ['B1', 'B5', 'lval', 'sasa', 'sasa_P', 'sasa_volume', "cone_angle", "p_int", "p_int_atom", "p_int_area", "pyr_val", "pyr_alpha", "qvbur_min", "qvbur_max", "qvtot_min", "qvtot_max", "max_delta_qvbur", "max_delta_qvtot", "ovbur_min", "ovbur_max", "ovtot_min", "ovtot_max", "vbur", "vtot", "near_vbur", "far_vbur", "near_vtot", "far_vtot"]

    for p in electronic_properties:
        if p in datagroups:
            feature_ref = ligand_data[molname][p+"_boltzmann"]
        else:
            feature_ref = ligand_data[molname][p]
        if feature_ref is not None:
            data_here=[]
            mask_here=[]
            for c_idx in range(0,n_conformers):
                if p in datagroups_vec:
                    x = data_here_confs["conf_%i"%(c_idx)]["electronic_properties"][p][p_idx]
                    if x is None:
                        #print("WARNING: found None in %s of conformer %i"%(p, c_idx))
                        #x = 0.0
                        data_here.append(x)
                    else:
                        mask_here.append(c_idx)
                        data_here.append(float(x))
                else:
                    x = data_here_confs["conf_%i"%(c_idx)]["electronic_properties"][p]
                    if x is None:
                        #print("WARNING: found None in %s of conformer %i"%(p, c_idx))
                        #x = 0.0
                        data_here.append(x)
                    else:
                        mask_here.append(c_idx)
                        data_here.append(float(x))
            mask_here=np.array(mask_here)
            if len(mask_here)!=len(weights_here):
                ligand_data[molname][p]=None
            else:
                feature_all = np.sum(np.array(data_here)*np.array(weights_here))
                feature_N = np.sum(np.array(data_here)[conformers_to_use]*weights_N)
                #print("%s:\naverage over all (%i): %.3f / %.3f\naverage over %i: %.3f"%(p, n_conformers, feature_all, feature_ref, N, feature_N))
            ligand_data[molname]["confdata"][p]=data_here


    for p in morfeus_parameters:
        if p in datagroups:
            feature_ref = ligand_data[molname][p+"_boltzmann"]
        else:
            feature_ref = ligand_data[molname][p]
        if feature_ref is not None:
            #print("read %s"%(p))
            data_here=[]
            mask_here=[]
            for c_idx in range(0,n_conformers):
                x = data_here_confs["conf_%i"%(c_idx)]["morfeus_parameters"][p]
                if x is None:
                    #print("WARNING: found None in %s of conformer %i"%(p, c_idx))
                    #x = 0.0
                    data_here.append(x)
                else:
                    mask_here.append(c_idx)
                    data_here.append(float(x))
            mask_here=np.array(mask_here)
            if len(mask_here)!=len(weights_here):
                ligand_data[molname][p]=None
            else:
                feature_all = np.sum(np.array(data_here)*np.array(weights_here))
                feature_N = np.sum(np.array(data_here)[conformers_to_use]*weights_N)
                #print("%s:\naverage over all (%i): %.3f / %.3f\naverage over %i: %.3f"%(p, n_conformers, feature_all, feature_ref, N, feature_N))
            ligand_data[molname]["confdata"][p]=data_here
        #else:
        #    print("WARNING: %s is None"%(p))


    outfilename="%s/%s_combined.yml"%(resultsdir, molname)
    outfile=open(outfilename,"w")
    outfile.write(yaml.dump(ligand_data[molname], default_flow_style=False))
    outfile.close()




def get_rotatable_bonds(smiles):
    m = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts('[*&!F&!Cl]-&!@[*&!F&!Cl]')
    single_bonds=m.GetSubstructMatches(patt)
    rotatable_bonds=[]
    for x in single_bonds:
        rotatable_bonds.append([x[0],x[1]])
    return(rotatable_bonds)

def get_num_bonds_P(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        print("WARNING: could not create mol from %s. assume P has 3 bonds."%(smiles))
        return(3)
    try:
        atoms=m.GetAtoms()
    except:
        print("WARNING: could not create mol from %s. assume P has 3 bonds."%(smiles))
        return(3)
    els=[a.GetSymbol() for a in atoms]
    if "P" in els:
        P_index=els.index("P")
    #elif "p" in smiles:
    #    P_index=smiles.index("p")
    else:
        exit("ERROR: no P found in smiles %s"%(smiles))
    P_atom=atoms[P_index]
    bonds=P_atom.GetBonds()
    num_bonds=0.0
    for bond in bonds:
        bondtype=bond.GetBondType()
        print("   ---   found a P bond: %s"%(str(bondtype)))
        if str(bondtype)=="SINGLE":
            num_bonds+=1.0
        elif str(bondtype)=="DOUBLE":
            num_bonds+=2.0
        elif str(bondtype)=="TRIPLE":
            num_bonds+=3.0
        elif str(bondtype)=="AROMATIC":
            num_bonds+=1.5
    if abs(num_bonds-round(num_bonds))>0.1:
        exit("ERROR: problem with bonds! %s"%(smiles))
    else:
        num_bonds=int(num_bonds)
        return(num_bonds)


def get_P_bond_indeces_of_ligand(coords, elements):
    bonds = get_bonds(coords, elements)
    #for bond in bonds:
    #    els=[elements[bond[0]],elements[bond[1]]]
    #    if "P" in els and "Pd" in els:
    #        if "P"==les[0]:
    #            P_index=bond[0]
    #        else:
    #            P_index=bond[1]
    #        break
    for P_index, element in enumerate(elements):
        if element=="P":
            break
    bond_indeces=[]
    for bond in bonds:
        idx1=bond[0]
        idx2=bond[1]
        if P_index==idx1:
            #print(idx1,idx2)
            bond_indeces.append(idx2)
        if P_index==idx2:
            #print(idx1,idx2)
            bond_indeces.append(idx1)
    return(P_index, bond_indeces)


def add_Hs_to_P(smiles, num_bonds_P):
    if "[P@" in smiles:
        return(smiles)
    if "P" in smiles:
        P_index=smiles.index("P")
        if P_index>0:
            if smiles[P_index-1]=="[":
                exit("ERROR: P is already in a square braket. cannot add explicit H's %s"%(smiles))
        if num_bonds_P==3:
            add="[P]"
        elif num_bonds_P==2:
            add="[PH]"
        elif num_bonds_P==1:
            add="[PH2]"
        elif num_bonds_P==0:
            add="[PH3]"
        else:
            add="[P]"
            print("WARNING: weird number of bonds (%i) for P in %s"%(num_bonds_P, smiles))


    elif "p" in smiles:
        P_index=smiles.index("p")
        if P_index>0:
            if smiles[P_index-1]=="[":
                exit("ERROR: P is already in a square braket. cannot add explicit H's %s"%(smiles))
        if num_bonds_P==3:
            add="p"
        elif num_bonds_P==2:
            add="[pH]"
        elif num_bonds_P==1:
            add="[pH2]"
        elif num_bonds_P==0:
            add="[pH3]"
        else:
            add="[p]"
            print("WARNING: weird number of bonds (%i) for p in %s"%(num_bonds_P, smiles))

    else:
        exit("ERROR: no P or p found in %s"%(smiles))

    p1=smiles[:P_index]
    p2=smiles[P_index+1:]

    smiles_new=p1+add+p2
    return(smiles_new)

def add_to_smiles(smiles, add):
    if "P" in smiles:
        P_index=smiles.index("P")
        if P_index>0:
            if smiles[P_index-1]=="[":
                print("   ---   found P in square brakets")
                if smiles[P_index+1]=="]":
                    P_index=P_index+1
                elif smiles[P_index+2]=="]":
                    P_index=P_index+2
                elif smiles[P_index+3]=="]":
                    P_index=P_index+3

    elif "p" in smiles:
        P_index=smiles.index("p")
        if P_index>0:
            if smiles[P_index-1]=="[":
                print("   ---   found p in square brakets")
                if smiles[P_index+1]=="]":
                    P_index=P_index+1
                elif smiles[P_index+2]=="]":
                    P_index=P_index+2
                elif smiles[P_index+3]=="]":
                    P_index=P_index+3
    else:
        print("no P or p found in %s"%(smiles))
    p1=smiles[:P_index+1]
    p2=smiles[P_index+1:]
    smiles_new=p1+"(%s)"%(add)+p2
    #print(smiles_new)
    #smiles_new=smiles_new.replace("Pd","X")
    #smiles_new=smiles_new.replace("P","[P]")
    #smiles_new=smiles_new.replace("X","Pd")
    #print(smiles_new)
    return(smiles_new)




def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def get_coords_from_smiles(smiles, suffix, conversion_method):
    if conversion_method=="any":
        to_try = ["rdkit", "molconvert", "obabel"]
    elif conversion_method=="rdkit":
        to_try = ["rdkit"]
    elif conversion_method=="molconvert":
        to_try = ["molconvert"]
    elif conversion_method=="obabel":
        to_try = ["obabel"]

    error=""
    for m in to_try:
        print("   ---   try to convert %s to 3D using %s (of %s)"%(smiles, m, str(to_try)))
        if m=="molconvert":

            if which("molconvert") != None:

                coords, elements = get_coords_from_smiles_marvin(smiles, suffix)
                if coords is None or elements is None:
                    error+=" molconvert_failed "
                    pass
                else:
                    if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                        print("   ---   conversion done with molconvert")
                        return(coords, elements)
                    else:
                        error+=" molconvert_mol_flat "
                        pass
                        #print("WARNING: molconvert produced a flat molecule. proceed with other methods (obabel or rdkit)")
            else:
                error+=" molconvert_not_available "

        if m=="obabel":
            if which("obabel") != None:
                #print("use obabel")
                coords, elements = get_coords_from_smiles_obabel(smiles, suffix)
                if coords is None or elements is None:
                    error+=" obabel_failed "
                    pass
                else:
                    if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                        print("   ---   conversion done with obabel")
                        return(coords, elements)
                    else:
                        error+=" obabel_failed "
                        pass

            else:
                error+=" obabel_not_available "

        if m=="rdkit":
            #print("use rdkit")
            coords, elements = get_coords_from_smiles_rdkit(smiles, suffix)
            if coords is None or elements is None:
                error+=" rdkit_failed "
                pass
            else:
                if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                    print("   ---   conversion done with rdkit")
                    return(coords, elements)
                else:
                    error+=" rdkit_failed "
                    pass

    exit("ERROR: NO 3D conversion worked: %s"%(error))

def get_coords_from_smiles_obabel(smiles, suffix):
    name=uuid.uuid4()

    if not os.path.exists("input_structures%s"%(suffix)):
        try:
            os.makedirs("input_structures%s"%(suffix))
        except:
            pass

    filename="input_structures%s/%s.xyz"%(suffix, name)
    os.system("obabel -:\"%s\" --gen3D -oxyz > %s"%(smiles, filename))
    if not os.path.exists(filename):
        return(None, None)
        #print("ERROR: could not convert %s to 3D using obabel. Exit!"%(smiles))
        #exit()

    coords, elements = readXYZ(filename)
    if len(coords)==0:
        return(None, None)
        #print("ERROR: could not convert %s to 3D using obabel. Exit!"%(smiles))
        #exit()
    os.system("rm %s"%(filename))
    return(coords, elements)


def get_coords_from_smiles_rdkit(smiles, suffix):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        return(None, None)
        #print("could not convert %s to rdkit molecule. Exit!"%(smiles))
        #exit()
    try:
        m = Chem.AddHs(m)
    except:
        return(None, None)
        #print("ERROR: could not add hydrogen to rdkit molecule of %s. Exit!"%(smiles))
        #exit()
    try:
        AllChem.EmbedMolecule(m)
    except:
        return(None, None)
        #print("ERROR: could not calculate 3D coordinates from rdkit molecule %s. Exit!"%(smiles))
        #exit()
    try:
        block=Chem.MolToMolBlock(m)
        blocklines=block.split("\n")
        coords=[]
        elements=[]
        for line in blocklines[4:]:
            if len(line.split())==4:
                break
            elements.append(line.split()[3])
            coords.append([float(line.split()[0]),float(line.split()[1]),float(line.split()[2])])
        coords=np.array(coords)
        mean = np.mean(coords, axis=0)
        distances = scsp.distance.cdist([mean],coords)[0]
        if np.max(distances)<0.1:
            return(None, None)
            #print("ERROR: something is wrong with rdkit molecule %s. Exit!"%(smiles))
            #print("%i\n"%(len(coords)))
            #for atomidx, atom in enumerate(coords):
            #    print("%s %f %f %f"%(elements[atomidx], atom[0], atom[1], atom[2]))
            #exit()
            
    except:
        return(None, None)
        #print("ERROR: could not read xyz coordinates from rdkit molecule %s. Exit!"%(smiles))
        #exit()
    return(coords, elements)




def get_coords_from_smiles_marvin(smiles, suffix):

    name=uuid.uuid4()

    if not os.path.exists("tempfiles%s"%(suffix)):
        try:
            os.makedirs("tempfiles%s"%(suffix))
        except:
            pass
    if not os.path.exists("input_structures%s"%(suffix)):
        try:
            os.makedirs("input_structures%s"%(suffix))
        except:
            pass

    outfile=open("tempfiles%s/%s.smi"%(suffix, name),"w")
    outfile.write("%s\n"%(smiles))
    outfile.close()

    path_here=os.getcwd()
    os.system("molconvert -2 mrv:+H %s/tempfiles%s/%s.smi > tempfiles%s/%s.mrv"%(path_here,suffix, name, suffix, name))
    filename="tempfiles%s/%s.mrv"%(suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi"%(suffix, name))
        return(None, None)
        #print("ERROR: could not convert %s to 2D (mrv) using marvin. Exit!"%(smiles))
        #exit()

    os.system("molconvert -3 xyz %s/tempfiles%s/%s.mrv > input_structures%s/%s.xyz"%(path_here, suffix, name, suffix, name))
    filename="input_structures%s/%s.xyz"%(suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv"%(suffix, name, suffix, name))
        return(None, None)
        #print("ERROR: could not convert %s to 3D (xyz) using marvin. Exit!"%(smiles))
        #exit()

    coords, elements = readXYZ(filename)
    if len(coords)==0:
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz"%(suffix, name, suffix, name, suffix, name))
        print("ERROR: could not convert %s to 3D (coords in empty) using marvin. Exit!"%(smiles))
        #return(None, None)
        #exit()
    os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz"%(suffix, name, suffix, name, suffix, name))
    return(coords, elements)



def remove_complex(coords, elements, smiles, settings):
    P_index = elements.index("P")
    mask, done = get_ligand_indeces(coords, elements, P_index, smiles, "Pd")
    if not done:
        return(None, None, done)
    coords_ligand=[]
    elements_ligand=[]
    for atomidx in mask:
        atom = coords[atomidx]
        elements_ligand.append(elements[atomidx])
        coords_ligand.append([atom[0],atom[1],atom[2]])
    coords_ligand=np.array(coords_ligand)
    return(coords_ligand, elements_ligand, True)

