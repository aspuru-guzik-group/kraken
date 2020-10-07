import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scli
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from rdkit import Chem

kcal_to_eV=0.0433641153
kB=8.6173303e-5 #eV/K
T=298.15
kBT=kB*T

def readXYZ(filename):
    infile=open(filename,"r")
    coords=[]
    elements=[]
    for line in infile.readlines()[2:]:
        elements.append(line.split()[0])
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    coords=np.array(coords)
    return coords,elements


def exportXYZ(coords,elements,filename):
    outfile=open(filename,"w")
    outfile.write("%i\n\n"%(len(elements)))
    for atomidx,atom in enumerate(coords):
        outfile.write("%s %f %f %f\n"%(elements[atomidx],atom[0],atom[1],atom[2]))
    outfile.close()




def exportXYZs(coords,elements,filename):
    outfile=open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx],atom[0],atom[1],atom[2]))
    outfile.close()


def get_rotatble_bonds(smiles):
    m = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts('[*&!F&!Cl]-&!@[*&!F&!Cl]')
    single_bonds=m.GetSubstructMatches(patt)
    return(single_bonds)




molnames=[]
for x in os.listdir("input_structures"):
    if ".xyz" in x:
        molname=x.replace(".xyz","")
        molnames.append(molname)
molnames=sorted(molnames)

#molnames=["P"]

smiles=[]
number_of_atoms=[]
time_all=[]
time_sterimol=[]
time_crest=[]
rotatable_bonds=[]
num_rotatable_bonds=[]
number_of_conformers=[]
L=[]
L_std=[]
B1=[]
B1_std=[]
B5=[]
B5_std=[]
sasa_val=[]
sasa_val_std=[]
sasa_val_P=[]
sasa_val_P_std=[]
sasa_volume=[]
sasa_volume_std=[]
sasa_volume_P=[]
sasa_volume_P_std=[]
cone_angle_val=[]
cone_angle_val_std=[]
muls=[]
muls_std=[]
wils=[]
wils_std=[]
global_electrophilicity_index=[]
global_electrophilicity_index_std=[]
fukui=[]
fukui_std=[]
dip_norm=[]
dip_norm_std=[]
alpha=[]
alpha_std=[]
EA_delta_SCC=[]
EA_delta_SCC_std=[]
HOMO_LUMO_gap=[]
HOMO_LUMO_gap_std=[]
IP_delta_SCC=[]
IP_delta_SCC_std=[]
nucleophilicity=[]
nucleophilicity_std=[]
boltzmann_weights=[]

HOMO_LUMO_gaps_all=[]
L_all=[]
B1_all=[]
B5_all=[]
failed_file=open("FAILED.dat","w")
for idx,molname in enumerate(molnames):
    if molname!="ligand_374600":
        continue
    outfilename="results_all/%s.yml"%(molname)
    if os.path.exists(outfilename):
        print("   ---   read molecule %s"%(outfilename))
        outfile=open(outfilename,"r")
        data_here=yaml.load(outfile)
        outfile.close()
        if data_here=="FAILED":
            print("   ---   molecule %s FAILED"%(outfilename))
            continue
        number_of_atoms.append(len(data_here["elements"]))
        num_rotatable_bonds.append(data_here["num_rotatable_bonds"])
        rotatable_bonds.append(data_here["rotatable_bonds"])
        number_of_conformers.append(data_here["number_of_conformers"])
        smiles.append(data_here["smiles"])
        time_all.append(data_here["time_all"])
        time_sterimol.append(data_here["time_sterimol"])
        time_crest.append(data_here["time_crest"])
        L.append(data_here["boltzmann_averaged_data"]["lval"])
        L_std.append(data_here["boltzmann_averaged_data"]["lval_std"])
        B1.append(data_here["boltzmann_averaged_data"]["B1"])
        B1_std.append(data_here["boltzmann_averaged_data"]["B1_std"])
        B5.append(data_here["boltzmann_averaged_data"]["B5"])
        B5_std.append(data_here["boltzmann_averaged_data"]["B5_std"])
        sasa_val.append(data_here["boltzmann_averaged_data"]["sasa"])
        sasa_val_std.append(data_here["boltzmann_averaged_data"]["sasa_std"])
        sasa_val_P.append(data_here["boltzmann_averaged_data"]["sasa_P"])
        sasa_val_P_std.append(data_here["boltzmann_averaged_data"]["sasa_P_std"])
        sasa_volume.append(data_here["boltzmann_averaged_data"]["sasa_volume"])
        sasa_volume_std.append(data_here["boltzmann_averaged_data"]["sasa_volume_std"])
        sasa_volume_P.append(data_here["boltzmann_averaged_data"]["sasa_volume_P"])
        sasa_volume_P_std.append(data_here["boltzmann_averaged_data"]["sasa_volume_P_std"])
        cone_angle_val.append(data_here["boltzmann_averaged_data"]["cone_angle"])
        cone_angle_val_std.append(data_here["boltzmann_averaged_data"]["cone_angle_std"])

        if L[-1]<1e-5:
            print("WARNING: L is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: L is 0 for ligand %s\n"%(molname))
        if B1[-1]<1e-5:
            print("WARNING: B1 is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: B1 is 0 for ligand %s\n"%(molname))
        if B5[-1]<1e-5:
            print("WARNING: B5 is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: B5 is 0 for ligand %s\n"%(molname))
        if sasa_val[-1]<1e-5:
            print("WARNING: sasa_val is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: sasa_val is 0 for ligand %s\n"%(molname))
        if sasa_val_P[-1]<1e-5:
            print("WARNING: sasa_val_P is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: sasa_val_P is 0 for ligand %s\n"%(molname))
        if sasa_volume[-1]<1e-5:
            print("WARNING: sasa_volume is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: sasa_volume is 0 for ligand %s\n"%(molname))
        if sasa_volume_P[-1]<1e-5:
            print("WARNING: sasa_volume_P is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: sasa_volume_P is 0 for ligand %s\n"%(molname))
        if cone_angle_val[-1]<1e-5:
            print("WARNING: cone_angle_val is 0 for ligand %s"%(molname))
            failed_file.write("WARNING: cone_angle_val is 0 for ligand %s\n"%(molname))

        p_idx=data_here["p_idx"]
        muls.append(data_here["boltzmann_averaged_data"]["muls"][p_idx])
        muls_std.append(data_here["boltzmann_averaged_data"]["muls_std"][p_idx])
        wils.append(data_here["boltzmann_averaged_data"]["wils"][p_idx])
        wils_std.append(data_here["boltzmann_averaged_data"]["wils_std"][p_idx])
        global_electrophilicity_index.append(data_here["boltzmann_averaged_data"]["global_electrophilicity_index"])
        global_electrophilicity_index_std.append(data_here["boltzmann_averaged_data"]["global_electrophilicity_index_std"])
        fukui.append(data_here["boltzmann_averaged_data"]["fukui"][p_idx])
        fukui_std.append(data_here["boltzmann_averaged_data"]["fukui_std"][p_idx])
        dip_norm.append(data_here["boltzmann_averaged_data"]["dip_norm"])
        dip_norm_std.append(data_here["boltzmann_averaged_data"]["dip_norm_std"])
        alpha.append(data_here["boltzmann_averaged_data"]["alpha"])
        alpha_std.append(data_here["boltzmann_averaged_data"]["alpha_std"])
        EA_delta_SCC.append(data_here["boltzmann_averaged_data"]["EA_delta_SCC"])
        EA_delta_SCC_std.append(data_here["boltzmann_averaged_data"]["EA_delta_SCC_std"])
        HOMO_LUMO_gap.append(data_here["boltzmann_averaged_data"]["HOMO_LUMO_gap"])
        HOMO_LUMO_gap_std.append(data_here["boltzmann_averaged_data"]["HOMO_LUMO_gap_std"])
        IP_delta_SCC.append(data_here["boltzmann_averaged_data"]["IP_delta_SCC"])
        IP_delta_SCC_std.append(data_here["boltzmann_averaged_data"]["IP_delta_SCC_std"])
        nucleophilicity.append(data_here["boltzmann_averaged_data"]["nucleophilicity"])
        nucleophilicity_std.append(data_here["boltzmann_averaged_data"]["nucleophilicity_std"])
        boltzmann_weights.append(data_here["boltzmann_weights"])


        HOMO_LUMO_gaps_all.append([])
        L_all.append([])
        B1_all.append([])
        B5_all.append([])
        filenames=["gaps_convergence","L_convergence","B1_convergence","B5_convergence"]
        do=False
        for fn in filenames:
            if not os.path.exists("property_convergence/data/%s_mol_%s.dat"%(fn, molname)):
                do=True
                break
        do=False
        if molname=="ligand_374600":
            do=True
        if do:
            outfilename_confs="results_all/%s_confs.yml"%(molname)

            outfile2=open(outfilename_confs,"r")
            data_here_confs=yaml.load(outfile2)
            outfile2.close()
            for c_idx in range(0,number_of_conformers[-1]):
                HOMO_LUMO_gaps_all[-1].append(data_here_confs["conf_%i"%(c_idx)]["electronic_properties"]["HOMO_LUMO_gap"])

            for c_idx in range(0,number_of_conformers[-1]):
                L_all[-1].append(data_here_confs["conf_%i"%(c_idx)]["sterimol_parameters"]["lval"])
            for c_idx in range(0,number_of_conformers[-1]):
                B1_all[-1].append(data_here_confs["conf_%i"%(c_idx)]["sterimol_parameters"]["B1"])
            for c_idx in range(0,number_of_conformers[-1]):
                B5_all[-1].append(data_here_confs["conf_%i"%(c_idx)]["sterimol_parameters"]["B5"])


    else:
        print("   ---   results for molecule %s missing"%(outfilename))

    #if (idx+1)%10==0 or (idx+1)==len(molnames):
    print("   ---   done with %i out of %i molecules"%(idx+1,len(molnames)))
exit()
'''
gaps_convergence=[]
gaps_convergence_std=[]
L_convergence=[]
L_convergence_std=[]
B1_convergence=[]
B1_convergence_std=[]
B5_convergence=[]
B5_convergence_std=[]
for molid in range(0,len(HOMO_LUMO_gaps_all)):
    
    if not os.path.exists("property_convergence/data/gaps_convergence_mol_%s.dat"%(molnames[molid])):
        n_confs=number_of_conformers[molid]
        weights=np.array(boltzmann_weights[molid])
        gap_mean=HOMO_LUMO_gap[molid]
        gap_std=HOMO_LUMO_gap_std[molid]
        gaps=np.array(HOMO_LUMO_gaps_all[molid])
        gaps_convergence_here=[]
        gaps_convergence_here_std=[]
        for i in range(1,n_confs+1):
            weights_new=np.copy(weights[:i])
            weights_new/=np.sum(weights_new)
            gap_new=np.average(gaps[:i], weights=weights_new, axis=0)
            gap_new_std = np.average((gaps[:i]-gap_new)**2.0, weights=weights_new, axis=0)**0.5
            gaps_convergence_here.append(gap_new)
            gaps_convergence_here_std.append(gap_new_std)
        gaps_convergence.append(gaps_convergence_here)
        gaps_convergence_std.append(gaps_convergence_here_std)
        np.savetxt("property_convergence/data/gaps_convergence_mol_%s.dat"%(molnames[molid]),gaps_convergence_here)
        np.savetxt("property_convergence/data/gaps_convergence_std_mol_%s.dat"%(molnames[molid]),gaps_convergence_here_std)

    if not os.path.exists("property_convergence/data/L_convergence_mol_%s.dat"%(molnames[molid])):
        n_confs=number_of_conformers[molid]
        weights=np.array(boltzmann_weights[molid])
        L=np.array(L_all[molid])
        L_convergence_here=[]
        L_convergence_here_std=[]
        for i in range(1,n_confs+1):
            weights_new=np.copy(weights[:i])
            weights_new/=np.sum(weights_new)
            L_new=np.average(L[:i], weights=weights_new, axis=0)
            L_new_std = np.average((L[:i]-L_new)**2.0, weights=weights_new, axis=0)**0.5
            L_convergence_here.append(L_new)
            L_convergence_here_std.append(L_new_std)
        L_convergence.append(L_convergence_here)
        L_convergence_std.append(L_convergence_here_std)
        np.savetxt("property_convergence/data/L_convergence_mol_%s.dat"%(molnames[molid]),L_convergence_here)
        np.savetxt("property_convergence/data/L_convergence_std_mol_%s.dat"%(molnames[molid]),L_convergence_here_std)

    if not os.path.exists("property_convergence/data/B1_convergence_mol_%s.dat"%(molnames[molid])):
        n_confs=number_of_conformers[molid]
        weights=np.array(boltzmann_weights[molid])
        B1=np.array(B1_all[molid])
        B1_convergence_here=[]
        B1_convergence_here_std=[]
        for i in range(1,n_confs+1):
            weights_new=np.copy(weights[:i])
            weights_new/=np.sum(weights_new)
            B1_new=np.average(B1[:i], weights=weights_new, axis=0)
            B1_new_std = np.average((B1[:i]-B1_new)**2.0, weights=weights_new, axis=0)**0.5
            B1_convergence_here.append(B1_new)
            B1_convergence_here_std.append(B1_new_std)
        B1_convergence.append(B1_convergence_here)
        B1_convergence_std.append(B1_convergence_here_std)
        np.savetxt("property_convergence/data/B1_convergence_mol_%s.dat"%(molnames[molid]),B1_convergence_here)
        np.savetxt("property_convergence/data/B1_convergence_std_mol_%s.dat"%(molnames[molid]),B1_convergence_here_std)


    if not os.path.exists("property_convergence/data/B5_convergence_mol_%s.dat"%(molnames[molid])):
        n_confs=number_of_conformers[molid]
        weights=np.array(boltzmann_weights[molid])
        B5=np.array(B5_all[molid])
        B5_convergence_here=[]
        B5_convergence_here_std=[]
        for i in range(1,n_confs+1):
            weights_new=np.copy(weights[:i])
            weights_new/=np.sum(weights_new)
            B5_new=np.average(B5[:i], weights=weights_new, axis=0)
            B5_new_std = np.average((B5[:i]-B5_new)**2.0, weights=weights_new, axis=0)**0.5
            B5_convergence_here.append(B5_new)
            B5_convergence_here_std.append(B5_new_std)
        B5_convergence.append(B5_convergence_here)
        B5_convergence_std.append(B5_convergence_here_std)
        np.savetxt("property_convergence/data/B5_convergence_mol_%s.dat"%(molnames[molid]),B5_convergence_here)
        np.savetxt("property_convergence/data/B5_convergence_std_mol_%s.dat"%(molnames[molid]),B5_convergence_here_std)



plt.figure()
for molid in range(0,len(HOMO_LUMO_gaps_all)):
    n_confs=number_of_conformers[molid]
    if n_confs>1:
        gaps_convergence_here=np.loadtxt("property_convergence/data/gaps_convergence_mol_%s.dat"%(molnames[molid]))
        gaps_convergence_here_std=np.loadtxt("property_convergence/data/gaps_convergence_std_mol_%s.dat"%(molnames[molid]))
        xs=range(1,n_confs+1)
        #print(len(xs))
        #print(len(gaps_convergence_here))
        #plt.errorbar(xs,gaps_convergence_here,yerr=gaps_convergence_here_std)
        plt.plot(xs,gaps_convergence_here,"-",linewidth=0.5)
plt.xlim([0,40])
plt.xlabel("Number of conformers")
plt.ylabel("HOMO LUMO gap [eV]")
plt.savefig("property_convergence/HOMO_LUMO_gaps.png",dpi=300)
plt.close()

plt.figure()
for molid in range(0,len(L_all)):
    n_confs=number_of_conformers[molid]
    if n_confs>1:
        L_convergence_here=np.loadtxt("property_convergence/data/L_convergence_mol_%s.dat"%(molnames[molid]))
        L_convergence_here_std=np.loadtxt("property_convergence/data/L_convergence_std_mol_%s.dat"%(molnames[molid]))
        xs=range(1,n_confs+1)
        #plt.errorbar(xs,L_convergence_here,yerr=L_convergence_here_std)
        plt.plot(xs,L_convergence_here,"-",linewidth=0.5)
plt.xlim([0,40])
plt.xlabel("Number of conformers")
plt.ylabel("L")
plt.savefig("property_convergence/L.png",dpi=300)
plt.close()

plt.figure()
for molid in range(0,len(B1_all)):
    n_confs=number_of_conformers[molid]
    if n_confs>1:
        B1_convergence_here=np.loadtxt("property_convergence/data/B1_convergence_mol_%s.dat"%(molnames[molid]))
        B1_convergence_here_std=np.loadtxt("property_convergence/data/B1_convergence_std_mol_%s.dat"%(molnames[molid]))
        xs=range(1,n_confs+1)
        #plt.errorbar(xs,B1_convergence_here,yerr=B1_convergence_here_std)
        plt.plot(xs,B1_convergence_here,"-",linewidth=0.5)
plt.xlim([0,40])
plt.xlabel("Number of conformers")
plt.ylabel("B1")
plt.savefig("property_convergence/B1.png",dpi=300)
plt.close()

plt.figure()
for molid in range(0,len(B5_all)):
    n_confs=number_of_conformers[molid]
    if n_confs>1:
        B5_convergence_here=np.loadtxt("property_convergence/data/B5_convergence_mol_%s.dat"%(molnames[molid]))
        B5_convergence_here_std=np.loadtxt("property_convergence/data/B5_convergence_std_mol_%s.dat"%(molnames[molid]))
        xs=range(1,n_confs+1)
        #plt.errorbar(xs,B5_convergence_here,yerr=B5_convergence_here_std)
        plt.plot(xs,B5_convergence_here,"-",linewidth=0.5)
plt.xlim([0,40])
plt.xlabel("Number of conformers")
plt.ylabel("B5")
plt.savefig("property_convergence/B5.png",dpi=300)
plt.close()
'''



number_of_atoms=np.array(number_of_atoms)
number_of_conformers=np.array(number_of_conformers)
time_all=np.array(time_all)
time_sterimol=np.array(time_sterimol)
time_crest=np.array(time_crest)


plt.figure()
plt.scatter(number_of_atoms,time_all/3600.0,c="blue",marker="o")
plt.xlabel("Number of atoms")
plt.ylabel("CPU time [h]")
plt.savefig("cpu_time_num_atoms.png",dpi=300)
plt.close()

plt.figure()
plt.scatter(np.array(num_rotatable_bonds),time_all/3600.0,c="blue",marker="o")
plt.xlabel("Number of rotatable bonds")
plt.ylabel("CPU time [h]")
plt.savefig("cpu_time_num_rotatable_bonds.png",dpi=300)
plt.close()


plt.figure()
plt.scatter(number_of_conformers,time_all/3600.0,c="blue",marker="o")
plt.xlabel("Number of conformers")
plt.ylabel("CPU time [h]")
plt.savefig("cpu_time_num_confs.png",dpi=300)
plt.close()

order=np.argsort(time_all)
xs=np.array(range(len(time_all)))

def estimate_DFT_time(num_atoms, number_of_conformers):
    time_for_50_atoms=5.0*60.0
    dft_time=number_of_conformers*time_for_50_atoms*(num_atoms/50.0)**3.0
    return(dft_time)
dft_time=estimate_DFT_time(number_of_atoms, np.array([min(40,i) for i in number_of_conformers]))

plt.figure()
plt.scatter(xs,time_all[order]/3600.0,c="k",marker="o",s=1,label="Total time")
plt.scatter(xs,time_sterimol[order]/3600.0,c="C1",marker="o",s=1,label="xtb")
plt.scatter(xs,time_crest[order]/3600.0,c="C2",marker="o",s=1,label="crest")
plt.scatter(xs,dft_time[order]/3600.0,c="C3",marker="o",s=1,label="DFT (estimated)")
plt.xlabel("Molecules ordered by total CPU time")
plt.ylabel("CPU time [h]")
plt.legend(loc="upper left")
plt.savefig("times_crest_xtb_DFT.png",dpi=300)
plt.close()


mean_cpu_time=np.mean(time_all)/3600.0
print("   ###   Mean cpu-time: %.1f hours"%(mean_cpu_time))
print("   ---   CPU time for 100.000 molecules:      %.0f CPU-hours"%(100000.0*mean_cpu_time))
print("   ---   Molecules done in 100.000 CPU-hours: %.0f molecules"%(100000.0/mean_cpu_time))



data=np.array([L, B1, B5, global_electrophilicity_index, alpha, muls]).T
columns=["L","B1","B5","Electrophilicity","Polarizability","Mulliken charge of P"]

df = pd.DataFrame(data=data,columns=columns)
sns_plot = sns.pairplot(df, diag_kind="kde")
sns_plot.savefig("L_B1_B5_pairplot.png", dpi=300)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(L, B1, B5, c="blue", marker="o")
ax.set_xlabel('L [$\AA$]')
ax.set_ylabel('B1 [$\AA$]')
ax.set_zlabel('B5 [$\AA$]')
plt.savefig("L_B1_B5.png",dpi=300)
plt.close()





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(global_electrophilicity_index, alpha, muls, c="blue", marker="o")
ax.set_xlabel('global electrophilicity index')
ax.set_ylabel('polarizability')
ax.set_zlabel('Mulliken charge of P')
plt.savefig("electrophilicity_alpha_mul_P.png",dpi=300)
plt.close()







