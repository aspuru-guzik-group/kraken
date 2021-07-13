# 201005: rename/restructure .yml files for consistency with xtb-level data
# 201006: in read_conformer() fix error message when log files are missing 

import os,re,itertools,time
#import pybel
#from openbabel import pybel
import numpy as np
import pandas as pd
import pathlib as pl
cwd = pl.Path.cwd()
import yaml
from yaml import CLoader as Loader
from yaml import CDumper as Dumper
from rdkit import Chem,Geometry
from rdkit.Chem import rdmolfiles, AllChem, rdMolAlign,rdmolops
from multiprocessing import Pool

import morfeus # Kjell Jorner
from PL_split_logs_201006 import split_log # TG
from PL_conformer_selection_200411 import mirror_mol, delete_element_from_rdkitmol, delete_haloalkane_halides # TG  #changed from PL_conformer_selection_201019 5/17/21 by EP
import PL_gaussian_properties_201021 as gp # TG
import vmin4 as vmin # TG/Iris Guo
import P_int_200916 as P_int # Robert Pollice (,TG(,ML))
# import PL_visvol as visvol # Ellyn Peters

# covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
# values for metals decreased by 10% according to Robert Paton's Sterimol implementation
rcov = {
"H": 0.32,"He": 0.46,"Li": 1.2,"Be": 0.94,"B": 0.77,"C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,"Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,"Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,"Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,"Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,"Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,"Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.08,"Ag": 1.15,"Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,"Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,"Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,"Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,"Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,"Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,"Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,"Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55
}

# some constants
R = 0.0019872036 #kcal mol^-1 K^-1
T = 298.15 #K
hartree_kcalmol = 627.50947 

periodictable = ["Bq","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub","Uut","Uuq","Uup","Uuh","Uus","Uuo","X"]

def get_conmat(elements, coords): 
    # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    # elements is a list of strings, coords is a numpy array or nested list of shape N_atoms x 3
    if type(coords) == list:
        coords = np.asarray(coords)
    natom = len(elements)
    #max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    for i in range(0,natom):
        if elements[i] not in rcov.keys():
            continue
        for iat in range(0,natom):
            if elements[iat] not in rcov.keys():
                continue
            if iat != i:
                dxyz = coords[iat]-coords[i]
                r = np.linalg.norm(dxyz)
                rco = rcov[elements[i]]+rcov[elements[iat]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0+np.math.exp(-k1*(rr-1.0)))
                if damp > 0.85: #check if threshold is good enough for general purpose
                    conmat[i,iat],conmat[iat,i] = 1,1
    return(conmat)
    
def add_valence(elements,coords,conmat,base_idx,add_element="Pd"):
    # Adds a valence to base so that the angle to the previous substituents is maximized and reorders the coordinate output for convenience
    # add_element: add any of the following elements:
    distpx = {"O":1.5,"Se":2.12,"Pd":2.28,"X":1.8} # typical bond distances to P
    if type(coords) == list:
        coords = np.asarray(coords)
    num_atoms = len(elements)
    coord_base = coords[base_idx]
    base_element = elements[base_idx]
    vec = np.array([0.0,0.0,0.0])
    bonded = []
    for atom in range(num_atoms):
        if conmat[base_idx][atom]:
            bonded.append(atom)
            vec += coord_base - coords[atom]
    coordox = distpx[add_element]*vec/np.linalg.norm(vec) + coord_base
    atoms = [x for x in range(num_atoms+1)]
    coords_temp = np.vstack((coords,coordox))
    if sum(get_conmat(elements+[add_element],coords_temp)[-1]) != 1.0:
        print("    Warning: possible collision!")
    # sort coordinates so that base is first, add_element is second, and the other atoms bonded to base are next
    elements_new = [base_element,add_element]+[elements[a] for a in bonded] + [a for i,a in enumerate(elements) if i not in [base_idx]+bonded]
    coords_new = np.vstack((coord_base, coordox, coords[bonded], coords[[i for i,a in enumerate(elements) if i not in [base_idx]+bonded]]))
    return(elements_new, coords_new)

def write_xyz(elements,coords,filename):
    with open(filename,"w") as f:
        f.write(f"{len(elements)}\n\n")
        for i,a in enumerate(elements):
            f.write(f"{a.title():>3} " + " ".join([f"{coords[i][j]:15f}" for j in range(3)]) + "\n")

def rmsd_matrix(conformers):
    molobjects = [rdmolfiles.MolFromMolFile(str(cwd/conformer/f"{conformer}_opt.sdf"),removeHs=False,strictParsing=False) for conformer in conformers]
    molobjects = [Chem.RemoveHs(mol) for mol in molobjects] # Remove all H: optional but speeds up RMSD calculation
    molobjects = [delete_haloalkane_halides(mol) for mol in molobjects] # Remove halides in perhaloalkyl moieties. Improves RMSD matching and timing
    molobjects_inv = [mirror_mol(mol) for mol in molobjects] # create mirror images of each conformer
    rmsd_mat = np.zeros((len(conformers),len(conformers)))
    for i,j in itertools.product(range(len(conformers)),range(len(conformers))):
        if i<j: continue
        if i==j: 
            rmsd_mat[i,j] = 1
        else:
            rmsd_mat[i,j] = min((rdMolAlign.GetBestRMS(molobjects[i],molobjects[j]),rdMolAlign.GetBestRMS(molobjects[i],molobjects_inv[j])))
            rmsd_mat[j,i] = rmsd_mat[i,j]
    return(rmsd_mat)

def dict_key_rmsd(candidate_pair):
    return float(rmsd_matrix(candidate_pair)[0,1])


# which energies to read from which log-file
energylogs = {
"e_dz":"freq",
"e_tz_gas":"nbo",
"e_tz_gas":"sp",
"e_tz_solv":"solv",
"e_tz_ra":"ra",
"e_tz_rc":"rc",
}

# which properties to read from which log-file
proplogs = {
"freq":["nimag","g","t"],
"sp"  :["dipole","homo","qpole","t"],
"ra"  :["homo","nbo","t"],
"rc"  :["homo","nbo","t"],
"nbo" :["nbo","nborbsP","t"],
"nmr" :["nmr","t"],
"efg" :["efg","nuesp","t"],
"solv":["ecds","t"],
}

# assign names to each descriptor
propoutput = {
"freq_g":      ["","g"],
"freq_nimag":  ["nimag"],
"sp_dipole": ["dipolemoment",],
"sp_homo":   ["fmo_e_homo","fmo_e_lumo","fmo_mu","fmo_eta","fmo_omega"],
"ra_homo":["somo_ra","","","",""],
"rc_homo":["somo_rc","","","",""],
"sp_qpole":  ["qpole_amp","qpoletens_xx","qpoletens_yy","qpoletens_zz"],
"nbo_nbo":    ["nbo_P"],
"ra_nbo": ["nbo_P_ra","spindens_P_ra"],
"rc_nbo": ["nbo_P_rc","spindens_P_rc"],
"nmr_nmr":    ["nmr_P","nmrtens_sxx_P","nmrtens_syy_P","nmrtens_szz_P",],
"efg_efg":    ["efg_amp_P","efgtens_xx_P","efgtens_yy_P","efgtens_zz_P"],
"efg_nuesp":  ["nuesp_P",],
"solv_ecds":  ["E_solv_cds"],
"nbo_dipole": ["dipolemoment",],
"nbo_homo":   ["fmo_e_homo","fmo_e_lumo","fmo_mu","fmo_eta","fmo_omega"],
"nbo_qpole":  ["qpole_amp","qpoletens_xx","qpoletens_yy","qpoletens_zz"],
}

boltzproperties = ['vmin_vmin','vmin_r','dipolemoment', 'fmo_e_homo', 'fmo_e_lumo', 'fmo_mu', 'fmo_eta', 'fmo_omega', 'somo_ra', 'somo_rc', 'qpole_amp', 'qpoletens_xx', 'qpoletens_yy', 'qpoletens_zz', 'nbo_P', 'nbo_P_ra', 'spindens_P_ra', 'nbo_P_rc', 'spindens_P_rc', 'nmr_P', 'nmrtens_sxx_P', 'nmrtens_syy_P', 'nmrtens_szz_P', 'efg_amp_P', 'efgtens_xx_P', 'efgtens_yy_P', 'efgtens_zz_P', 'nuesp_P', 'E_solv_cds', 'nbo_lp_P_percent_s', 'nbo_lp_P_occ', 'nbo_lp_P_e', 'nbo_bd_e_max', 'nbo_bd_e_avg', 'nbo_bds_e_min', 'nbo_bds_e_avg', 'nbo_bd_occ_min', 'nbo_bd_occ_avg', 'nbo_bds_occ_max', 'nbo_bds_occ_avg', 'E_solv_total', 'E_solv_elstat', 'E_oxidation', 'E_reduction', 'fukui_p', 'fukui_m', 'pyr_P', 'pyr_alpha', 'vbur_vbur', 'vbur_vtot', 'vbur_ratio_vbur_vtot', 'vbur_qvbur_min', 'vbur_qvbur_max', 'vbur_qvtot_min', 'vbur_qvtot_max', 'vbur_max_delta_qvbur', 'vbur_max_delta_qvtot', 'vbur_ovbur_min', 'vbur_ovbur_max', 'vbur_ovtot_min', 'vbur_ovtot_max', 'vbur_near_vbur', 'vbur_far_vbur', 'vbur_near_vtot', 'vbur_far_vtot', 'sterimol_B1', 'sterimol_B5', 'sterimol_L', 'sterimol_burB1', 'sterimol_burB5', 'sterimol_burL',"Pint_P_int","Pint_dP","Pint_P_min","Pint_P_max","volume","surface_area","sphericity"] # "vv_total_visible_volume","vv_proximal_visible_volume","vv_distal_visible_volume","vv_ratio_visible_total","vv_ratio_proxvis_total",

mmproperties = ['dipolemoment', 'qpole_amp', 'qpoletens_xx', 'qpoletens_yy', 'qpoletens_zz', 'pyr_P', 'pyr_alpha', 'vbur_vbur', 'vbur_vtot', 'vbur_qvbur_min', 'vbur_qvbur_max', 'vbur_qvtot_min', 'vbur_qvtot_max', 'vbur_max_delta_qvbur', 'vbur_max_delta_qvtot', 'vbur_ovbur_min', 'vbur_ovbur_max', 'vbur_ovtot_min', 'vbur_ovtot_max', 'vbur_near_vbur', 'vbur_far_vbur', 'vbur_near_vtot', 'vbur_far_vtot', 'sterimol_B1', 'sterimol_B5', 'sterimol_L', 'sterimol_burB1', 'sterimol_burB5', 'sterimol_burL'] # ,"vv_total_visible_volume","vv_proximal_visible_volume","vv_distal_visible_volume","vv_ratio_visible_total","vv_ratio_proxvis_total",

Pintresults = ["Pint_P_int","Pint_dP","Pint_P_min","Pint_P_max","volume","surface_area","sphericity"]

def morfeus_properties(elements,coordinates,confdata):
    # Morfeus: Sterimol, Vbur, pyr
    morfdict = {}
    if "pyr_P" not in confdata.keys() and confdata["p_val"] == 3:
        # Pyramidalization - two equivalent measurments P and alpha    
        pyr = morfeus.Pyramidalization(elements=elements,coordinates=coordinates,atom_index=1,excluded_atoms=[2]) # remove Pd
        morfdict["pyr_P"] = float(pyr.P)
        morfdict["pyr_alpha"] = float(pyr.alpha)

    if "vbur_vbur" not in confdata.keys():
        #Buried volume - get quadrant volumes and distal volume 
        # iterate through P-substituents, aligning the quadrants paralell to each once (= xz_plane definition)
        # Metal/point of reference should be 2.28 A away from P
        # z_axis_atoms: P  
        # xz_plane_atoms: each of the substituents once
        # keep lowest and highest quadrant and octant volume across all three orientations of the coordinate system
        # keep highest difference of any neighboring quadrant volume
        # keep volume in each of the two hemispheres 

        qvbur_all = np.array([])
        qvdist_all = np.array([])
        qvtot_all = np.array([])
        max_delta_qvbur_all = []
        max_delta_qvtot_all = []
        ovbur_all = np.array([])
        ovtot_all = np.array([])
        
        for i in range(3):#confdata["p_val"]):  
            bv = morfeus.BuriedVolume(elements,coordinates,2,excluded_atoms=[2],z_axis_atoms=[1],xz_plane_atoms=[3+i]) 
            bv.octant_analysis()
            bv.compute_distal_volume(method="buried_volume",octants=True)

            vbur = bv.buried_volume   # these are identical for each iteration
            vdist = bv.distal_volume  #  
            vtot = vbur + vdist       #  

            qvbur = np.asarray(list(bv.quadrants["buried_volume"].values()))
            qvdist = np.asarray(list(bv.quadrants["distal_volume"].values()))
            qvtot = qvbur + qvdist

            qvbur_all = np.append(qvbur_all,qvbur)
            qvtot_all = np.append(qvtot_all,qvtot)

            max_delta_qvbur_all.append(max([abs(qvbur[j]-qvbur[j-1]) for j in range(4)]))
            max_delta_qvtot_all.append(max([abs(qvtot[j]-qvtot[j-1]) for j in range(4)]))

            ovbur = np.asarray(list(bv.octants["buried_volume"].values()))
            ovdist = np.asarray(list(bv.octants["distal_volume"].values()))
            ovtot = ovbur + ovdist

            ovbur_all = np.append(ovbur_all,ovbur)
            ovtot_all = np.append(ovtot_all,ovtot)

            near_vbur = ovbur[4:].sum()   # these are identical for each iteration
            far_vbur = ovbur[:4].sum()    # 
            near_vtot = ovtot[4:].sum()   # 
            far_vtot = ovtot[:4].sum()    # 

        morfdict["vbur_vbur"] = vbur
        morfdict["vbur_vtot"] = float(vtot)
        morfdict["vbur_ratio_vbur_vtot"] = float(vbur/vtot)

        morfdict["vbur_qvbur_min"] = float(min(qvbur_all))
        morfdict["vbur_qvbur_max"] = float(max(qvbur_all))
        morfdict["vbur_qvtot_min"] = float(min(qvtot_all))
        morfdict["vbur_qvtot_max"] = float(max(qvtot_all))

        morfdict["vbur_max_delta_qvbur"] = float(max(max_delta_qvbur_all))
        morfdict["vbur_max_delta_qvtot"] = float(max(max_delta_qvtot_all))

        morfdict["vbur_ovbur_min"] = float(min(ovbur_all))
        morfdict["vbur_ovbur_max"] = float(max(ovbur_all))
        morfdict["vbur_ovtot_min"] = float(min(ovtot_all))
        morfdict["vbur_ovtot_max"] = float(max(ovtot_all))

        morfdict["vbur_near_vbur"] = float(near_vbur)
        morfdict["vbur_far_vbur"]  = float(far_vbur)
        morfdict["vbur_near_vtot"] = float(near_vtot)
        morfdict["vbur_far_vtot"]  = float(far_vtot)

    if "sterimol_B1" not in confdata.keys():
        # Sterimol
        # for Sterimol values matching Rob Paton's implementation:
        patonradii = morfeus.helpers.get_radii(elements, radii_type="bondi")
        patonradii = np.array(patonradii)
        patonradii[patonradii == 1.2] = 1.09

        sterimol = morfeus.Sterimol(elements, coordinates, 2, 1, radii=patonradii, n_rot_vectors=3600)
        morfdict["sterimol_B1"] = float(sterimol.B_1_value)
        morfdict["sterimol_B5"] = float(sterimol.B_5_value)
        morfdict["sterimol_L"]  = float(sterimol.L_value)
        # buried Sterimol
        sterimol_bur = morfeus.Sterimol(elements, coordinates, 2, 1,calculate=False,radii=patonradii, n_rot_vectors=3600)
        sterimol_bur.bury(sphere_radius=5.5,method="delete",radii_scale=0.5) 
        # sterimol.bury(sphere_radius=4.5,method="delete",radii_scale=1) 
        morfdict["sterimol_burB1"] = float(sterimol_bur.B_1_value)
        morfdict["sterimol_burB5"] = float(sterimol_bur.B_5_value)
        morfdict["sterimol_burL"]  = float(sterimol_bur.L_value)

    return(morfdict)

def gp_properties(ligand,conformer,p_idx):
    # reads gaussian log files
    gpdict = {} 
    gpdict["properties"] = {}
    contents = {
        "streams":{},
        "filecont":{},
    }
    # read energies
    for e,log in energylogs.items():
        contents["streams"][log] = gp.get_outstreams(cwd/conformer/f"{conformer}_{log}.log")
        if contents["streams"][log] == "failed or incomplete job":
            return({"error":True})
        else:
            gpdict[e] = gp.get_e_hf(contents["streams"][log])
    gpdict["error"] = False
    # going through each log file, get the relevant properties
    for log in proplogs.keys():
        contents["filecont"][log] = gp.get_filecont(cwd/conformer/f"{conformer}_{log}.log")
        for prop in proplogs[log]:
            gpresults = gp.jobtypes[prop][0](contents[gp.jobtypes[prop][1]][log],p_idx)
            if prop == "nborbsP": # NBO orbital analysis returns a dictionary with the proper labels 
                gpdict["properties"].update(gpresults)
            elif prop == "t": # subjob time
                gpdict[f"{log}_t"] = gpresults
            elif prop in ["e_dz","g","e_tz_gas","e_tz_solv","e_tz_ra","e_tz_rc","nimag"]:
                gpdict.update({propoutput[f"{log}_{prop}"][i]: float(gpresults[i]) for i in range(len(gpresults))})
            else: # all other functions return a list. This is assigned into a dict with proper names here
                gpdict["properties"].update({propoutput[f"{log}_{prop}"][i]: float(gpresults[i]) for i in range(len(gpresults))})

    gpdict["g_tz_gas"]  = gpdict["g"] - gpdict["e_dz"] + gpdict["e_tz_gas"] # in Hartree
    gpdict["g_tz_solv"] = gpdict["g"] - gpdict["e_dz"] + gpdict["e_tz_solv"] # in Hartree
    gpdict["properties"]["E_solv_total"] = (gpdict["e_tz_solv"] - gpdict["e_tz_gas"]) * hartree_kcalmol # in kcal/mol
    gpdict["properties"]["E_solv_elstat"] = gpdict["properties"]["E_solv_total"] - gpdict["properties"]["E_solv_cds"] # in kcal/mol
    gpdict["properties"]["E_oxidation"] = gpdict["e_tz_rc"] - gpdict["e_tz_gas"] # in Hartree
    gpdict["properties"]["E_reduction"] = gpdict["e_tz_ra"] - gpdict["e_tz_gas"] # in Hartree
    gpdict["properties"]["fukui_p"] = gpdict["properties"]["nbo_P"]-gpdict["properties"]["nbo_P_ra"] # fukui electrophilicity 
    gpdict["properties"]["fukui_m"] = gpdict["properties"]["nbo_P_rc"]-gpdict["properties"]["nbo_P"] # fukui nucleophilicity
    gpdict["t_total"] = sum([gpdict[f"{log}_t"] for log in proplogs.keys()])
    if "" in gpdict.keys():
        del gpdict[""]
    if "" in gpdict["properties"].keys():
        del gpdict["properties"][""]
    return(gpdict)

def read_conformer(cwd, ligand, conformer): # cwd: pathlib path of current working directory. ligand: 0-digit ligand ID. conformer: full name of the conformer (including the ID at the beginnig)
    confdata = {}
    errors = []
    checklogs = [cwd/conformer/f"{conformer}_{l}.log" for l in proplogs.keys() if not (cwd/conformer/f"{conformer}_{l}.log").exists()]
    if len(checklogs) != 0:
        #! log this as a conformer-level error
        err = f"Missing Gaussian log files, flagged in read_conformer: {','.join([chkl.name for chkl in checklogs])}"
        errors.append(err)
        print(f"{ligand};{conformer};{err}")
        with open(cwd/f"{ligand}_errors.txt","a") as f:
            f.write(f"{ligand};{conformer};{err}\n")
        confdata["error"] = True
        return(confdata,errors)

    if "elements_pd" not in confdata.keys():
        # mol = next(pybel.readfile("g09",str(cwd/conformer/f"{conformer}_nbo.log")))
        #mol = next(pybel.readfile("g09",str(cwd/conformer/f"{conformer}_opt.log")))
        #elements = [periodictable[a.atomicnum] for a in mol.atoms]
        #coordinates = [list(a.coords) for a in mol.atoms]
        #coordinates_a = np.array([a.coords for a in mol.atoms])

        def read_gaussian_logfile(fn):
            time0=time.time()
            read=False
            for line in open(fn,"r"):
                if read:
                    if "---" in line and len(elements)>0:
                        read=False
                if read:
                    if "X" not in line and "---" not in line:
                        atomnum = int(line.split()[1])
                        #print(line.replace("\n",""))
                        #print(atomnum)
                        el = periodictable[atomnum]
                        elements.append(el)
                        coordinates.append([float(line.split()[3]),float(line.split()[4]), float(line.split()[5])])
                if "Coordinates (Angstroms)" in line:
                    coordinates, elements = [], []
                    read=True
            time1=time.time()
            print("gaussian log parser done in %.2f seconds"%(time1-time0))
            return(coordinates, elements)

        coordinates, elements = read_gaussian_logfile(str(cwd/conformer/f"{conformer}_opt.log"))
        coordinates_a = np.array(coordinates)

        conmat = get_conmat(elements,coordinates_a)
        p_idx = [i for i in range(len(elements)) if elements[i] == "P" and sum(conmat[i]) <= 3][0] # this removes quaternary P (phosphonium, phosphate etc) but allows for P with 2 substituents (phosphabenzene, phosphaimine etc). Can we be sure that we never have more than one non-quaternary P(III)? 
        elements_pd, coordinates_pd = add_valence(elements,coordinates,conmat,p_idx,add_element="Pd") # Add "Pd" at the reference position in the P-lone pair region
        if not (cwd/conformer/f"{conformer}_opt_Pd.xyz").exists():
            #out = pybel.Outputfile("xyz",str(cwd/conformer/f"{conformer}_opt.xyz"))
            #out.write(mol)
            #out.close()
            write_xyz(elements, coordinates, cwd/conformer/f"{conformer}_opt.xyz")
            #out = pybel.Outputfile("sdf",str(cwd/conformer/f"{conformer}_opt.sdf"))
            #out.write(mol)
            #out.close()
            os.system("obabel -ixyz %s -osdf >> %s"%(str(cwd/conformer/f"{conformer}_opt.xyz"), str(cwd/conformer/f"{conformer}_opt.sdf")))
            write_xyz(elements_pd,coordinates_pd,cwd/conformer/f"{conformer}_opt_Pd.xyz")
        confdata["coords"] = coordinates
        confdata["coords_pd"] = coordinates_pd.tolist()
        confdata["elements"] = elements
        confdata["elements_pd"] = elements_pd
        confdata["conmat"] = conmat.tolist()
        confdata["p_idx"] = p_idx
        confdata["p_val"] = int(sum(conmat[p_idx])) # how many substituents at P

    confdata["properties"] = {}
    ## get properties
    # gp_properties: everything that can be read from the Gaussian log files (most electronic properties)
    confdata.update(gp_properties(ligand,conformer,confdata["p_idx"]))
    if confdata["error"]:
        #! log this as a conformer-level error
        err = "Error in the Gaussian computations, flagged in read_conformer, please check log files."
        errors.append(err)
        print(f"{ligand};{conformer};{err}")
        with open(cwd/f"{ligand}_errors.txt","a") as f:
            f.write(f"{ligand};{conformer};{err}\n")
        with open(cwd/conformer/f"{conformer}_data.yml","w") as f:
            yaml.dump(confdata,f,Dumper=Dumper)
        return(confdata,errors)

    if confdata["nimag"] != 0:
        #! log this as a conformer-level error
        err = f"Number of imaginary frequencies: {confdata['nimag']}."
        errors.append(err)
        print(f"{ligand};{conformer};{err}")
        with open(cwd/f"{ligand}_errors.txt","a") as f:
            f.write(f"{ligand};{conformer};{err}\n")
        with open(cwd/conformer/f"{conformer}_data.yml","w") as f:
            yaml.dump(confdata,f,Dumper=Dumper)
        confdata["error"] = True
        return(confdata,errors)

    # morfeus: properties that use the geometry/steric properties
    confdata["properties"].update(morfeus_properties(confdata["elements_pd"],confdata["coords_pd"],confdata))

    # # P_int
    # if "Pint_P_int" not in confdata.keys():
    #     confdata.update(P_int.P_int_main(name=conformer,directory=cwd/conformer))
    # read results
    disp = "d3"
    pint_read = P_int.read_dedout(cwd/conformer,conformer,disp)+P_int.read_multiwfnout(cwd/conformer,conformer)+P_int.read_disp(cwd/conformer,conformer,disp)
    confdata["properties"].update({Pintresults[i]:float(pint_read[i]) for i in range(7)})
    
    # V_min
    try:
        if "vmin_vmin" not in confdata.keys():
            vminob = vmin.get_vmin(f"{conformer}.fchk",str(cwd/conformer)+"/",True)
            confdata["properties"]["vmin_vmin"] = float(vminob.v_min)
            confdata["properties"]["vmin_r"] = float(vminob.r_min)
    except:
        err = f"Vmin FileNotFoundError."
        errors.append(err)
        print(f"{ligand};{conformer};{err}")
        with open(cwd/f"{ligand}_errors.txt","a") as f:
            f.write(f"{ligand};{conformer};{err}\n")
        confdata["error"] = True

    # visvol
    # if "vv_total_visible_volume" not in confdata.keys():
    #     confdata.update(visvol.get_vis_vol(cwd/conformer/f"{conformer}_opt_Pd.xyz",radii_type = 'rcov',prox_cutoff = 3.5,ignore_H = 0,write_results = 1, plot = 0))

    with open(cwd/conformer/f"{conformer}_data.yml","w") as f:
        yaml.dump(confdata,f,Dumper=Dumper)

    return(confdata,errors)

def read_ligand(cwd, ligand, conformers, liganddata = {}): # cwd is the ligand-level directory
    status = {"ligandlevel": [],}
    if len(liganddata.keys()) == 0:
        if (cwd/f"{ligand}_data.yml").exists():
            with open(cwd/f"{ligand}_data.yml","r") as f:
                liganddata = yaml.load(f,Loader=Loader)
            if (cwd/f"{ligand}_confdata.yml").exists():
                with open(cwd/f"{ligand}_confdata.yml","r") as f:
                    liganddata["confdata"] = yaml.load(f,Loader=Loader)
        
        else:
            liganddata = {
                "conformers_all": conformers,  
                "conformers": conformers.copy(),        # Duplicates and computations with errors (including nimag=1) will be removed from this list 
                "number_of_conformers": len(conformers),
                "removed_duplicates": [],
                "confdata": {},#{c:{} for c in conformers},
                "boltzmann_averaged_data": {},
                "min_data": {},
                "max_data": {},
                "delta_data": {},
                "vburminconf_data": {},
            }

    newconfs = 0
    for conformer in conformers:
        if conformer in liganddata["removed_duplicates"]:
            continue

        print(conformer)
        if conformer in liganddata["confdata"].keys():
            pass
        elif (cwd/conformer/f"{conformer}_data.yml").exists():
            with open(cwd/conformer/f"{conformer}_data.yml","r") as f:
                liganddata["confdata"][conformer] = yaml.load(f,Loader=Loader)
            newconfs += 1
        else:
            print("read conformer data")
            liganddata["confdata"][conformer],status[conformer] = read_conformer(cwd, ligand, conformer) # returns the dictionary with the conformer data and a list with errors
            newconfs += 1

    if newconfs > 0:
        # error, NIMAG removal
        liganddata["conformers_w_error"] = [conformer for conformer in liganddata["conformers"] if liganddata["confdata"][conformer]["error"]]
        liganddata["conformers"] = [c for c in liganddata["conformers"] if c not in liganddata["conformers_w_error"]]
        liganddata["number_of_conformers"] = len(liganddata["conformers"])
        energies = ["e_dz","g","e_tz_gas","g_tz_gas","e_tz_solv","g_tz_solv"]
        liganddata["energies"] = {}
        liganddata["relative_energies"] = {}
        for e in energies:
            liganddata["energies"][e] = {conformer: liganddata["confdata"][conformer][e] for conformer in liganddata["conformers"]}
            liganddata[e+"_min"] = min(liganddata["energies"][e].values())
            liganddata[e+"_minconf"] = list(liganddata["energies"][e].keys())[np.argmin(list(liganddata["energies"][e].values()))]
            liganddata["relative_energies"][e+"_rel"] = {conformer: (liganddata["energies"][e][conformer]-liganddata[e+"_min"])*hartree_kcalmol for conformer in liganddata["conformers"]}

        # erel_df = pd.DataFrame(np.array([list(liganddata[e+"_rel"].values()) for e in energies]).T ,columns=energies,index=liganddata["conformers"] )
        erel_df = pd.DataFrame([liganddata["relative_energies"][e+"_rel"] for e in energies],index=energies).T
        #liganddata["relative_energies_df"] = erel_df
        liganddata["relative_energies_dict"] = erel_df.to_dict()

        # Find duplicates: 
        #  1) find pairs of conformers that are within E_rel < 0.1 kcal/mol (relative energies seem to be much more reliable than relative free energies)
        #  2) check these pairs to also have RMSD < 0.2 A 
        #  3) Remove the conformer with higher relative free energy
        duplicates_candidates = [(i,j) for i,j in itertools.combinations(liganddata["conformers"],2) if abs(erel_df["e_dz"].loc[i] - erel_df["e_dz"].loc[j]) < 0.1]
        try:
            # Throw a name error here if you wanna only run the except
            cores = max(os.cpu_count() - 2, 1)
            with Pool(cores) as p:
                values = p.map(dict_key_rmsd, duplicates_candidates)

            liganddata["rmsd_candidates"] = {key: value for key, value in zip(duplicates_candidates, values)}

            # The less cool, non-parallel way
            #liganddata["rmsd_candidates"] = {candidate_pair: float(rmsd_matrix(candidate_pair)[0,1]) for candidate_pair in duplicates_candidates} # keep all RMSD for potential debugging
            liganddata["duplicates"] = [candidate_pair for candidate_pair in liganddata["rmsd_candidates"] if liganddata["rmsd_candidates"][candidate_pair] < 0.2] 
        
        except: # RDkit failed to generate Mol objects and thus could not compute RMSD, or some of the internal structures in those mol files are different despite actually being the same. Default to duplicate detection based on dipole moment and chemical shift similarity
            #! log this on ligand level for double-checking
            err = "Warning: RDKit error at duplicate RMSD testing. Please double check."
            status["ligandlevel"].append(err)
            print(f"{ligand};ligandlevel;{err}")
            with open(cwd/f"{ligand}_errors.txt","a") as f:
                f.write(f"{ligand};ligandlevel;{err}\n")
            
            dipole_candidates = set([(i,j) for i,j in duplicates_candidates if abs(liganddata["confdata"][i]["properties"]["dipolemoment"] - liganddata["confdata"][j]["properties"]["dipolemoment"]) < 0.025])
            nmr_candidates = set([(i,j) for i,j in duplicates_candidates if abs(liganddata["confdata"][i]["properties"]["nmr_P"] - liganddata["confdata"][j]["properties"]["nmr_P"]) < 0.1])
            liganddata["duplicates"] = sorted(dipole_candidates & nmr_candidates)

        liganddata["removed_duplicates"] = [erel_df.loc[list(pair)]["g_tz_gas"].idxmax() for pair in liganddata["duplicates"]]
        liganddata["conformers"] = [c for c in liganddata["conformers"] if c not in liganddata["removed_duplicates"]]
        liganddata["number_of_conformers"] = len(liganddata["conformers"])

        # Boltzmann averaging 
        #boltzfacs = {conformer: np.exp(-liganddata["relative_energies_df"]["g_tz_gas"].loc[conformer]/(R*T)) for conformer in liganddata["conformers"]}
        boltzfacs = {conformer: np.exp(-erel_df["g_tz_gas"].loc[conformer]/(R*T)) for conformer in liganddata["conformers"]}

        Q = sum(boltzfacs.values())
        liganddata["boltzmann_weights"] = {conformer: float(boltzfacs[conformer]/Q) for conformer in liganddata["conformers"] } # probability
        for prop in boltzproperties:
            confsmissingprop = [conf for conf in liganddata["conformers"] if prop not in liganddata["confdata"][conf]["properties"].keys()]
            if len(confsmissingprop) == 0:
                liganddata["boltzmann_averaged_data"][prop] = sum([liganddata["boltzmann_weights"][conf] * liganddata["confdata"][conf]["properties"][prop] for conf in liganddata["conformers"]])
            else: # if a single conformer is missing a property value, set Boltzmann-average to None
                #! log this as a ligand-level error with prop and confsmissingprop
                err = f"Warning: {len(confsmissingprop)}/{len(liganddata['conformers'])} conformers missing values for property {prop}: {','.join(confsmissingprop)}."
                status["ligandlevel"].append(err)
                print(f"{ligand};ligandlevel;{err}")
                with open(cwd/f"{ligand}_errors.txt","a") as f:
                    f.write(f"{ligand};ligandlevel;{err}\n")
                liganddata["boltzmann_averaged_data"][prop] = None
                continue

        # "Condensed" properties
        liganddata["vburminconf"] = liganddata["conformers"][np.argmin([liganddata["confdata"][conf]["properties"]["vbur_vbur"] for conf in liganddata["conformers"]])]
        for prop in mmproperties:
            proplist = [liganddata["confdata"][conf]["properties"][prop] for conf in liganddata["conformers"] if prop in liganddata["confdata"][conf]["properties"].keys()] 
            # if a single conformer is missing a property value, still perform min/max analysis (Boltzmann-average will be None to indicate missing value(s))
            # if all confs are missing this prop, set min/max/delta to None
            if len(proplist) == 0:
                liganddata["min_data"][prop] = None
                liganddata["max_data"][prop] = None
                liganddata["delta_data"][prop] = None
                liganddata["vburminconf_data"][prop] = None
            else:
                liganddata["min_data"][prop] = min(proplist)
                liganddata["max_data"][prop] = max(proplist)
                liganddata["delta_data"][prop] = liganddata["max_data"][prop] - liganddata["min_data"][prop]
                liganddata["vburminconf_data"][prop] = liganddata["confdata"][liganddata["vburminconf"]]["properties"][prop]
        
        liganddata["time_all"] = sum([liganddata["confdata"][conf]["t_total"] for conf in liganddata["conformers_all"] if "t_total" in liganddata["confdata"][conf].keys()])

        with open(cwd/f"{ligand}_data.yml","w") as f:
            yaml.dump({k:v for k,v in liganddata.items() if k != "confdata"},f,Dumper=Dumper)
        with open(cwd/f"{ligand}_confdata.yml","w") as f:
            yaml.dump(liganddata["confdata"],f,Dumper=Dumper)
        erel_df.to_csv(cwd/f"{ligand}_relative_energies.csv",sep=";")

    return(liganddata,status)


def main_split_logs(cwd, ligand):
    if not (cwd/"ERR").exists():
        (cwd/"ERR").mkdir()
    # if not (cwd/"done").exists():
    #     (cwd/"done").mkdir()  
    conformers = [i.name for i in (cwd/ligand).iterdir() if i.is_dir()]
    conformers_good = []
    for conformer in conformers:
        logs = [i.name for i in (cwd/ligand/conformer).rglob("*.log")]
        if f"{conformer}.log" in logs and f"{conformer}_opt.log" not in logs:
            status = split_log(ligand, conformer)
            if status != "Error":
                #(cwd/ligand/conformer/f"{conformer}.log").rename(cwd/f"done/{conformer}.log")
                conformers_good.append(conformer)
    return(conformers_good)

if __name__ == '__main__': 
    starttime_all = time.time()

    ligname = re.compile("[0-9]{8}")  
    ligands = sorted([i.name for i in cwd.iterdir() if (ligname.match(i.name) and i.is_dir())])
    conformers = {ligand: [i.name for i in (cwd/ligand).iterdir() if i.is_dir()] for ligand in ligands}

    if not (cwd/"ERR").exists():
        (cwd/"ERR").mkdir()
    if not (cwd/"done").exists():
        (cwd/"done").mkdir()  

    for ligand in ligands:
        for conformer in conformers[ligand]:
            logs = [i.name for i in (cwd/ligand/conformer).rglob("*.log")]
            if f"{conformer}.log" in logs and f"{conformer}_opt.log" not in logs:
                status = split_log(ligand,conformer)
                if status != "Error":
                    (cwd/ligand/conformer/f"{conformer}.log").rename(cwd/f"done/{conformer}.log")

    
    if (cwd/"allligands_data.yml").exists():
        with open(cwd/"allligands_data.yml","r") as f:
            allliganddata = yaml.load(f,Loader=Loader)
    else:
        allliganddata = {}

    for ligand in ligands:
        print(ligand)
        print(conformers[ligand])
        if ligand in allliganddata.keys():
            allliganddata[ligand],status = read_ligand(cwd,ligand,conformers[ligand],allliganddata[ligand])
        else:
            allliganddata[ligand],status = read_ligand(cwd,ligand,conformers[ligand])

    with open(cwd/"allligands_data.yml","w") as f:
        yaml.dump(allliganddata,f,Dumper=Dumper)

    variants = ["boltz","min","max","delta","vburminconf"]
    columns = [i+"_boltz" for i in boltzproperties if i not in mmproperties] + [f"{i}_{j}" for i,j in itertools.product(mmproperties,variants)]# + ["t_total","number_of_conformers"] 
    df = pd.DataFrame(columns = columns,index = ligands)
    for l in ligands:
        for c in columns:
            print(allliganddata[l]["properties"])
            exit()
            df.loc[l][c] = allliganddata[l]["properties"][c]
    df["t_total"] = [allliganddata[l]["t_total"] for l in ligands]
    df["number_of_conformers"] = [allliganddata[l]["number_of_conformers"] for l in ligands]
    df.to_csv("allligands_data.csv",sep=";")

    print(f"All done. Total time: {round((time.time()-starttime_all),2)} sec")
            
