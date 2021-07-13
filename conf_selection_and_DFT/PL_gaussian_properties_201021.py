#201021 fixed nbo orbital occupancy descriptor

import os,re,subprocess
import numpy as np

float_pattern = re.compile(r"[-+]?\d*\.\d+")
float_pattern_p = re.compile(r"[-+]?\d*\.\d*")

def get_outstreams(filename): # gets the compressed stream information at the end of a Gaussian job
    streams = []
    starts,ends = [],[]
    error = "failed or incomplete job" # default unless "normal termination" is in file
    with open(filename,"r") as f:
        filecont = f.readlines()
    for i in range(len(filecont)):
        if "1\\1\\" in filecont[i]:
            starts.append(i)
        if "@" in filecont[i]:
            ends.append(i)
        if "Normal termination" in filecont[i]:
            error = ""
    if len(starts) != len(ends) or len(starts) == 0 or error != "": 
        error = "failed or incomplete job"
        return(error)
    for i in range(len(starts)):
        stream = "".join([line.strip() for line in filecont[starts[i]:ends[i]+1]]).split("\\")
        streams.append(stream)
    return(streams)

def get_filecont(filename): # gets the entire job output
     # default unless "normal termination" is in file
    with open(filename,"r") as f:
        filecont = f.readlines()
    for line in filecont[-10:]:
        if "Normal termination" in line:
            return(filecont)
    return(["failed or incomplete job"])
    
# def get_geom(streams): # extracts the geometry from the compressed stream - input orientation!
#     geom = []
#     for item in streams[-1][16:]:
#         if item == "":
#             break
#         geom.append([item.split(",")[0],float(item.split(",")[-3]),float(item.split(",")[-2]),float(item.split(",")[-1])])
#     return(geom)
    
# def get_geom_std(filecont):
#     geom = []
#     for ind,line in enumerate(filecont[::-1]):
#         if geom_pattern.search(line):
#             for line_ in filecont[-ind+4:]:
#                 if len(line_.split()) < 6:
#                     break
#                 ls = line_.split()
#                 geom.append([periodictable[int(ls[1])]] + [float(i) for i in ls[-3:]])
#             return(geom)

# def get_specdata(atoms,prop): # general purpose function for NMR, NBO, possibly other data with similar output structure      
#     propout = ""
#     #print(atoms)
#     for atom in atoms:
#         if atom.isdigit():
#             a = int(atom)-1
#             if a <= len(prop):
#                 propout += prop[a][0]+str(a+1)+";"+prop[a][1]+";"
#             else: continue
#         elif atom.isalpha():    
#             for a in range(len(prop)):
#                 if prop[a][0] == atom:
#                     #print(prop[a][0])
#                     propout += prop[a][0]+str(a+1)+";"+prop[a][1]+";"
#         else: continue
#     return(propout)
    
def get_time(filecont,blank):    # cpu and wall time
    cputime_pattern = "Job cpu time:"
    walltime_pattern = "Elapsed time:"
    
    cputime,walltime = 0,0
    for line in filecont[::-1]:
        if re.search(cputime_pattern,line):
            lsplt = str.split(line)
            cputime += float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
        if re.search(walltime_pattern,line):
            lsplt = str.split(line)
            walltime += float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
    return(cputime)

def get_e_hf(streams): # electronic energy of all subjobs
    for item in streams[-1]:
        if "HF=" in item:
            e_hf = float(re.findall(float_pattern,item)[0])
            return(e_hf)
    return(None)

def get_homolumo(filecont,blank): # homo,lumo energies and derived values of last job in file  
    homo_pattern = "Alpha  occ. eigenvalues"
    osmo_pattern = "Beta  occ. eigenvalues"
    lumo = 100
    for i in range(len(filecont)-1,0,-1):
        if re.search(osmo_pattern,filecont[i]) and lumo == 100:
            lumo = float(re.findall(float_pattern,filecont[i+1])[0])
        if re.search(homo_pattern,filecont[i]):
            homo = float(re.findall(float_pattern,filecont[i])[-1]) # in open shell systems, this is the SOMO
            lumo = min((lumo,float(re.findall(float_pattern,filecont[i+1])[0])))
            mu =  (homo+lumo)/2 # chemical potential / negative of molecular electronegativity
            eta = lumo-homo     # hardness/softness
            omega = mu**2/(2*eta) # electrophilicity index
            return([homo,lumo,mu,eta,omega])
    return([None])

def get_enthalpies(filecont,blank): # Gets thermochemical data from Freq jobs
    zero_pattern = "zero-point Energies"
    for i in range(len(filecont)-1):
        if re.search(zero_pattern,filecont[i]):
            e_zpe = float(re.search(float_pattern,filecont[i])[0])
            h     = float(re.search(float_pattern,filecont[i+2])[0])
            g     = float(re.search(float_pattern,filecont[i+3])[0])
            e_hf  = e_zpe - float(re.search(float_pattern,filecont[i-4])[0])
            return([e_hf,g]) # don't need the other values right now
    return([None])

# def get_g_scaled(log,blank):
#     error = "Some problem"
#     try:
#         std = subprocess.run("python -m goodvibes "+log+".log",shell=True, stdout=subprocess.PIPE)
#         g_scaled = float_pattern.findall(str(std))[-1] # finds the last float in the output. this is the scaled G
#         try: 
#             os.remove("Goodvibes_output.dat")
#         except:
#             pass
#         return(g_scaled)
#     except:
#         return(None)

def get_nimag(streams,blank):
    for stream in streams:
        for item in stream:
            if "NImag" in item:
                nimag = int(item[6:])
                return([nimag])
    return([None])
    
def get_nbo(filecont,query): 
    """Return the NBO partial charge of atom 'query'."""
    nbo_pattern = "Summary of Natural Population Analysis:"
    for i in range(len(filecont)-1):
        if re.search(nbo_pattern,filecont[i]):
            nboline = re.findall(float_pattern,filecont[i+6+query])
            nbo = float(nboline[0])
            if len(nboline) == 6:
                spindens = float(nboline[-1])
                return([nbo,spindens])
            else:
                return([nbo])
    return([None])
    
# def get_nbo_orbs(filecont,query): # pop=nbo
#     """Return all bonding, antibonding and lone pair (NB)orbitals that atom 'query' is involved in as a dictionary."""
#     nbo_pattern = "Natural Bond Orbitals (Summary)"
#     orbitals = {}
#     for i in range(len(filecont)):
#         if re.search(nbo_pattern,filecont[i],re.IGNORECASE):      
#             for j in range(i,len(filecont)):
#                 if str(query+1) in " ".join(re.findall("([A-Z][a-z]? *[0-9]+)",filecont[j])).split() and ("LP" in filecont[j] or "BD" in filecont[j]):
#                     orbital_desc = [re.search("[0-9]+\.[A-Z\*(0-9 ]+\)",filecont[j])[0] + " " +" - ".join(re.findall("([A-Z][a-z]? *[0-9]+)",filecont[j]))]
#                     orbitals[orbital_desc] = [float(x) for x in re.findall(float_pattern,filecont[j])] # orbital occupancy and energy
#             return(orbitals)
#     return(None)

def get_nbo_orbsP(filecont,query): # pop=nbo
    """Return all bonding, antibonding and lone pair (NB)orbitals that atom 'query' is involved in as a dictionary."""
    nbo_sum_pattern = "Natural Bond Orbitals"# (Summary)"
    nbo_an_pattern  = "NATURAL BOND ORBITAL ANALYSIS:"
    orbitals = {}
    for i in range(len(filecont)):

        if re.search(nbo_an_pattern,filecont[i],re.IGNORECASE):
            for j in range(i+10,len(filecont)):
                if " LP ( 1) P" in filecont[j]:
                    lp_percent_s = float(float_pattern.findall(filecont[j])[1])
                    break
        if re.search(nbo_sum_pattern,filecont[i],re.IGNORECASE):      
            for j in range(i,len(filecont)):
                if str(query+1) in " ".join(re.findall("([A-Z][a-z]? *[0-9]+)",filecont[j])).split() and ("LP" in filecont[j] or "BD" in filecont[j]):
                    orbital_desc = re.search("[0-9]+\.[A-Z\*(0-9 ]+\)",filecont[j])[0] + " " +" - ".join(re.findall("([A-Z][a-z]? *[0-9]+)",filecont[j]))
                    orbitals[orbital_desc] = [float(x) for x in re.findall(float_pattern,filecont[j])] # orbital occupancy and energy
            bd_occ  = [orbitals[i][0] for i in orbitals.keys() if "BD (" in i]
            bds_occ = [orbitals[i][0] for i in orbitals.keys() if "BD*(" in i]
            bd_e    = [orbitals[i][1] for i in orbitals.keys() if "BD (" in i]
            bds_e   = [orbitals[i][1] for i in orbitals.keys() if "BD*(" in i]
            results = {
            "nbo_lp_P_percent_s": lp_percent_s,
            "nbo_lp_P_occ": [orbitals[i] for i in orbitals.keys() if "LP (" in i][0][0],
            "nbo_lp_P_e"  : [orbitals[i] for i in orbitals.keys() if "LP (" in i][0][1],
            "nbo_bd_e_max": max(bd_e),
            "nbo_bd_e_avg": np.mean(bd_e),
            "nbo_bds_e_min": min(bds_e),
            "nbo_bds_e_avg": np.mean(bds_e),
            "nbo_bd_occ_min": min(bd_occ),
            "nbo_bd_occ_avg": np.mean(bd_occ),
            "nbo_bds_occ_max": max(bds_occ),
            "nbo_bds_occ_avg": np.mean(bds_occ),
            }
            results["nbo_delta_lp_P_bds"] = results["nbo_bds_e_min"] - results["nbo_lp_P_e"]   
            for k,v in results.items():
                results[k] = float(v)
            return(results)
    return(None)

def get_nmr(filecont,query): # nmr=giao
    """Return the isotropic chemical shift and chemical shift anisotropy tensor eigenvalues of atom 'query'."""
    nmrstart_pattern = "SCF GIAO Magnetic shielding tensor"
    for i in range(len(filecont)-1):
        if re.search(nmrstart_pattern,filecont[i]):
            shift_s = float(re.findall(float_pattern,filecont[i+1+query*5])[0])
            anisotropy_ev = [float(x) for x in re.findall(float_pattern,filecont[i+5+query*5])]
            return([shift_s]+anisotropy_ev)            
    return([None])

def get_dipole(streams,blank):
    """Return the absolute dipole moment in Debye."""
    for item in streams[-1]:
        if "Dipole" in item:
            d_vec = [float(i) for i in re.findall(float_pattern_p,item)]
            d_abs = np.linalg.norm(d_vec) * 2.541746  # conversion from Bohr-electron to Debye
            return([d_abs])
    return([None])       

def get_quadrupole(streams,blank):    
    """Return a 4-member list with the xx,yy,zz eigenvalues and the amplitude of the quadrupole moment tensor."""
    for item in streams[-1]:
        if "Quadrupole" in item:
            q = [float(i) for i in re.findall(float_pattern_p,item)]
            q_comps = np.array(([q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]))
            q_diag = np.linalg.eig(q_comps)[0]
            q_ampl = np.linalg.norm(q_diag)
            q_results = [q_ampl,np.max(q_diag),-(np.max(q_diag)+np.min(q_diag)),np.min(q_diag)]
            return(q_results)
    return([None])

def get_efg(filecont,query):
    """Return a 4-member list with the xx,yy,zz eigenvalues and the amplitude of the Electric Field Gradient tensor for the atom 'query'."""
    efg_pattern = "Center         ---- Electric Field Gradient ----"
    for i in range(len(filecont)-1):
        if re.search(efg_pattern,filecont[i]) and "Eigenvalues" in filecont[i+2]:
            efg_ev = np.asarray([float(i) for i in filecont[i+query+4].split()[2:5]])
            efg_ampl = np.linalg.norm(efg_ev)
            return(list(np.hstack((efg_ampl,efg_ev))))
    return([None])

def get_nuesp(filecont,query):
    """Return the electrostatic potential at atom 'query'."""
    nuesp_pat = "Electrostatic Properties (Atomic Units)"
    for i in range(len(filecont)-1):
        if nuesp_pat in filecont[i]:
            nuesp = float(filecont[i+6+query].split()[2])
            return([nuesp]) 
    return([None])

def get_edisp(filecont,blank):
    """Return the empirical dispersion energy correction to the nuclear repulsion."""
    edisp_pattern = "Nuclear repulsion after empirical dispersion term"
    erep_pattern = "nuclear repulsion energy"

    disp = 0
    for i in range(len(filecont)-1,1,-1):
        if re.search(edisp_pattern,filecont[i]):
            e_rep_disp = float(filecont[i].split()[-2]) # hartree 
            disp = 1
        if re.search(erep_pattern,filecont[i]) and disp == 1:
            e_rep_nodisp = float(filecont[i].split()[-2]) # hartree 
            e_disp = (e_rep_disp - e_rep_nodisp) * 627.50947 # kcal/mol
            return([e_disp])
    return([None])         

def get_ecds(filecont,blank):
    """Return the CDS part of solvation energy."""
    cds_pattern = "SMD-CDS (non-electrostatic) energy"
    for i in range(len(filecont)-1):
        if cds_pattern in filecont[i]:
            e_cds = float(filecont[i].split()[-1]) # comes in kcal/mol
            return([e_cds])
    return([None])

jobtypes = {
"e":          [get_e_hf,"streams"],
"homo":       [get_homolumo,"filecont"],
"g":          [get_enthalpies,"filecont"],
"nimag":      [get_nimag,"streams"],
"nbo":        [get_nbo,"filecont"],
"nborbsP":    [get_nbo_orbsP,"filecont"],
"nmr":        [get_nmr,"filecont"],
"dipole":     [get_dipole,"streams"],
"qpole":      [get_quadrupole,"streams"],
"efg":        [get_efg,"filecont"],
"nuesp":      [get_nuesp,"filecont"],
"edisp":      [get_edisp,"filecont"],
"ecds":       [get_ecds,"filecont"],    
"t":          [get_time,"filecont"],
}



# "gscaled":    [get_g_scaled,return_log],
# "nborbsP":    [get_nbo_orbsP,get_filecont],
# "nmrtens":    [get_nmrtens,get_filecont],
# "planeangle": [get_planeangle,get_outstreams],
# "dihedral":   [get_dihedrals,get_outstreams],
# "angle":      [get_angles,get_outstreams],
# "dist":       [get_distances,get_outstreams],
# "sterimol":   [run_sterimol2,return_log],
# "polar":      [get_polarizability,get_filecont],
# "surface":    [get_cavity,get_filecont],
# "ir":         [get_ir,get_filecont],
# "X":          [get_method,get_outstreams],
# "vers":       [get_version,get_outstreams],
# "fukui":      [get_fukui,get_filecont],
# "hirsh":      [get_hirsh,get_filecont], 
# "chelpg":     [get_chelpg,get_filecont], 
# "pyr":        [get_pyramidalization,get_outstreams],
# "route":      [get_route,get_outstreams],
# "nbohomo":    [get_nbohomolumo,get_filecont],
# "vbur":       [get_vbur,get_outstreams]
