#!/usr/bin/env python
import os
import sys
import numpy as np
import subprocess
import shlex
import time
import yaml
import copy
from rdkit import Chem
# own stuff
import utils
import geo_utils


if __name__ == "__main__":
    if len(sys.argv)==5 and not "-idx" in sys.argv:
        add_Ni_CO_3 = None
        if sys.argv[1]=="-name":
            molname=sys.argv[2]
        elif sys.argv[3]=="-name":
            molname=sys.argv[4]

        conversion_method = "any"
        generate_xyz = False
        if sys.argv[1]=="-xyz":
            filename=sys.argv[2].replace(".xyz","")
            if os.path.exists("smiles_codes/%s.smi"%(filename)):
                smiles=open("smiles_codes/%s.smi"%(filename),"r").readlines()[0].split()[0]
            else:
                smiles="not available"
        elif sys.argv[3]=="-xyz":
            filename=sys.argv[4].replace(".xyz","")
            if os.path.exists("smiles_codes/%s.smi"%(filename)):
                smiles=open("smiles_codes/%s.smi"%(filename),"r").readlines()[0].split()[0]
            else:
                smiles="not available"

        elif sys.argv[1]=="-smi":
            smiles=sys.argv[2]
            generate_xyz=True
        elif sys.argv[3]=="-smi":
            smiles=sys.argv[4]
            generate_xyz=True

    elif len(sys.argv)==5 and "-idx" in sys.argv:
        idx=int(sys.argv[2])
        listfilename=sys.argv[3]
        if sys.argv[4] == "Ni":
            add_Ni_CO_3 = True
        elif sys.argv[4] == "noNi":
            add_Ni_CO_3 = False
        else:
            exit("weird argument: %s"%(sys.argv[3]))
        generate_xyz=True
        if ".csv" in listfilename:
            for lineidx, line in enumerate(open(listfilename,"r")):
                if lineidx>0 and (lineidx-1)==idx:
                    if line.count(",")==3:
                        smiles=line.split(",")[2]
                        molname="%s_%s"%(line.split(",")[0], line.split(",")[1])
                        conversion_flag=int(line.split(",")[3].replace("\n",""))
                    elif line.count(",")==2:
                        smiles=line.split(",")[1]
                        molname="%s"%(line.split(",")[0])
                        conversion_flag=int(line.split(",")[2].replace("\n",""))
                    else:
                        exit("ERROR: unexpected number of commas in %s. exit."%(line))
                    break
        else:
            for lineidx, line in enumerate(open(listfilename,"r")):
                if lineidx==idx:
                    smiles=line.split()[1]
                    molname=line.split()[0]
                    conversion_flag=3
                    break
        #0	RDkit
        #1	Chemaxon
        #2	manual
        if conversion_flag == 2:
            generate_xyz = False
            conversion_method = None
            filename=molname
        elif conversion_flag == 0:
            conversion_method = "rdkit"
        elif conversion_flag == 1:
            conversion_method = "molconvert"
        elif conversion_flag == 3:
            conversion_method = "obabel"
        elif conversion_flag == 4:
            conversion_method = "any"
            

    else:
        print("Usage: run_kraken.py -smi/xyz smilescode/filename -name calcname")
        exit()
    startdir=os.getcwd()

    if add_Ni_CO_3 is not None:
        if add_Ni_CO_3:
            suffix="_Ni"
        else:
            suffix="_noNi"
    else:
        suffix=""



    print("   ###   Start with molecule %s"%(molname))
    print("   ---   Read settings")
    settings={"max_E":6.0,
              "max_p":0.1,
              "OMP_NUM_THREADS":3,
              "MKL_NUM_THREADS":2,
              "dummy_distance":1.5,
              "remove_scratch": True,
              "reduce_output": True,
              "add_Pd_Cl2": False,
              "add_Pd_Cl2_PH3": False,
              "add_Ni_CO_3": False
              }
    if os.path.exists("settings%s.yml"%(suffix)):
        infile=open("settings%s.yml"%(suffix),"r")
        settings_own=yaml.load(infile, Loader=yaml.FullLoader)
        infile.close()
        for x in settings_own:
            print("   ---   replace %s with %s"%(x,str(settings_own[x])))
            settings[x]=settings_own[x]
    else:
        print("   ---   no settings%s.yml found. use default settings"%(suffix))

    if add_Ni_CO_3 is not None:
        if add_Ni_CO_3:
            settings["add_Ni_CO_3"]=True
            settings["add_Pd_Cl2"]=False
            settings["add_Pd_Cl2_PH3"]=False
        else:
            settings["add_Ni_CO_3"]=False
            settings["add_Pd_Cl2"]=False
            settings["add_Pd_Cl2_PH3"]=False


    calcdir = "calculations%s"%(suffix)
    if not os.path.exists(calcdir):
        try:
            os.makedirs(calcdir)
        except:
            pass

    inputdir = "input_structures%s"%(suffix)
    if not os.path.exists(inputdir):
        try:
            os.makedirs(inputdir)
        except:
            pass

    resultsdir = "results_all%s"%(suffix)
    if not os.path.exists(resultsdir):
        try:
            os.makedirs(resultsdir)
        except:
            pass

    debugdir = "debugging%s"%(suffix)
    if not os.path.exists(debugdir):
        try:
            os.makedirs(debugdir)
        except:
            pass

    outfilename="%s/%s/%s.yml"%(startdir, resultsdir, molname)
    if os.path.exists(outfilename):
        print("Final results found: %s/%s/%s.yml . Exit."%(startdir, resultsdir, molname))
        exit()

    time1=time.time()
    # read or generate the coordinates
    moldir="%s%s"%(molname, suffix)



    #smiles="[P-]([H])([H])([H])[Pd+]([Cl])[Cl]"
    #smiles2 = utils.sanitize_smiles(smiles)
    #print(smiles)
    #print(smiles2)
    #coords, elements = utils.get_coords_from_smiles(smiles2, suffix, conversion_method)
    #utils.exportXYZ(coords,elements,"test.xyz")
    #exit()

    if generate_xyz:

        # check if smiles has to be extended

        if settings["add_Ni_CO_3"] and "Ni" not in smiles:
            metal_char="Ni"
            settings["metal_char"]=metal_char
            #print(smiles)
            num_bonds_P = utils.get_num_bonds_P(smiles)
            #print(num_bonds_P)
            smiles_Hs = utils.add_Hs_to_P(smiles, num_bonds_P)
            #print(smiles_Hs)
            #smiles = utils.add_to_smiles(smiles,"[Pd@SP1]([Cl])([PH3])[Cl]")
            #smiles = utils.add_to_smiles(smiles,"[Pd]([Cl])([PH3])[Cl]")
            #spacer_smiles="[Pd]([As+](F)(F)F)([As+](F)(F)F)[As+](F)(F)F"
            spacer_smiles="[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]"
            smiles_incl_spacer = utils.add_to_smiles(smiles_Hs, spacer_smiles)
            #print(smiles_incl_spacer)
            coords_ligand_complex, elements_ligand_complex = utils.get_coords_from_smiles(smiles_incl_spacer, suffix, conversion_method)


            #utils.exportXYZ(coords_ligand_complex, elements_ligand_complex, "test.xyz")
            num_atoms_with_fake_complex = len(coords_ligand_complex)
            print("   ---   length of the molecule after adding the fake complex: %i"%(num_atoms_with_fake_complex))
            coords_ligand, elements_ligand, done = utils.remove_complex(coords_ligand_complex, elements_ligand_complex, smiles, settings)
            if not done:
                utils.exportXYZ(coords_ligand_complex, elements_ligand_complex, "%s/%s.xyz"%(debugdir, molname))
                exit()
            #utils.exportXYZ(coords_ligand, elements_ligand, "test2.xyz")
            num_atoms_without_fake_complex = len(coords_ligand)
            print("   ---   length of the molecule after removing the fake complex: %i"%(num_atoms_without_fake_complex))
            difference = num_atoms_with_fake_complex - num_atoms_without_fake_complex
            if difference!=6:
                print("ERROR: number of removed atoms is %i, but should be 6 for Pd(Cl)5"%(difference))
                utils.exportXYZ(coords_ligand_complex, elements_ligand_complex, "%s/%s.xyz"%(debugdir, molname))
                exit()

            P_index, bond_indeces = utils.get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)
            if len(bond_indeces)!=3:
                print("WARNING, weird number of P-bonds before adding the complex: %i %s"%(len(bond_indeces), smiles))
            #print("bond_indeces", bond_indeces)
            direction=np.zeros((3))
            for bond_index in bond_indeces:
                direction+=(coords_ligand[bond_index]-coords_ligand[P_index])
            direction/=(-np.linalg.norm(direction))
            coords_ligand=np.array(coords_ligand.tolist()+[(coords_ligand[P_index]+2.25*direction).tolist()])
            elements_ligand.append(metal_char)
            match_pd_ind=len(elements_ligand)-1
            match_p_idx=P_index
            # replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, rotate_third_axis=True)
            #coords_pd, elements_pd, pd_idx, p_idx = geo_utils.get_Pd_NH3_Cl_Cl()
            #metal_char="Pd"
            coords_pd, elements_pd, pd_idx, p_idx = geo_utils.get_Ni_CO_3()
            success, coords, elements = geo_utils.replace(coords_pd, elements_pd, coords_ligand, elements_ligand, pd_idx, p_idx, match_pd_ind, match_p_idx, smiles, rotate_third_axis=True)
            if elements==None:
                exit("ERROR: elements is None %s"%(smiles))
            if len(elements)==0:
                exit("ERROR: elements is empty %s"%(smiles))

            #print(coords[0])
            coords, elements = utils.xtb_opt(coords, elements, smiles)
            #print(coords[0])
            coords, elements = utils.xtb_opt(coords, elements, smiles)
            #print(coords[0])
            coords, elements = utils.xtb_opt(coords, elements, smiles)
            #print(coords[0])

            P_index=elements.index("P")
            #As_index=elements.index("As")
            #elements[As_index]="N"
            settings["P_index"]=P_index
            if not success:
                exit("ERROR: Pd addition did not work. Exit.")


        elif settings["add_Pd_Cl2_PH3"] and "Pd" not in smiles:
            metal_char="Pd"
            settings["metal_char"]=metal_char
            #print(smiles)
            num_bonds_P = utils.get_num_bonds_P(smiles)
            #print(num_bonds_P)
            smiles_Hs = utils.add_Hs_to_P(smiles, num_bonds_P)
            #print(smiles_Hs)
            #smiles = utils.add_to_smiles(smiles,"[Pd@SP1]([Cl])([PH3])[Cl]")
            #smiles = utils.add_to_smiles(smiles,"[Pd]([Cl])([PH3])[Cl]")
            #spacer_smiles="[Pd]([As+](F)(F)F)([As+](F)(F)F)[As+](F)(F)F"
            spacer_smiles="[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]"
            smiles_incl_spacer = utils.add_to_smiles(smiles_Hs, spacer_smiles)

            coords_ligand_complex, elements_ligand_complex = utils.get_coords_from_smiles(smiles_incl_spacer, suffix, conversion_method)

            #utils.exportXYZ(coords_ligand_complex, elements_ligand_complex, "test.xyz")
            print("   ---   length of the molecule after adding the fake complex: %i"%(len(coords_ligand_complex)))
            coords_ligand, elements_ligand = utils.remove_complex(coords_ligand_complex, elements_ligand_complex, smiles, settings)
            #utils.exportXYZ(coords_ligand, elements_ligand, "test2.xyz")
            print("   ---   length of the molecule after removing the fake complex: %i"%(len(coords_ligand)))


            P_index, bond_indeces = utils.get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)
            if len(bond_indeces)!=3:
                print("WARNING, weird number of P-bonds before adding the complex: %i %s"%(len(bond_indeces), smiles))
            #print("bond_indeces", bond_indeces)
            direction=np.zeros((3))
            for bond_index in bond_indeces:
                direction+=(coords_ligand[bond_index]-coords_ligand[P_index])
            direction/=(-np.linalg.norm(direction))
            coords_ligand=np.array(coords_ligand.tolist()+[(coords_ligand[P_index]+2.25*direction).tolist()])
            elements_ligand.append("Pd")
            match_pd_ind=len(elements_ligand)-1
            match_p_idx=P_index
            # replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, rotate_third_axis=True)
            coords_pd, elements_pd, pd_idx, p_idx = geo_utils.get_Pd_NH3_Cl_Cl()
            metal_char="Pd"
            success, coords, elements = geo_utils.replace(coords_pd, elements_pd, coords_ligand, elements_ligand, pd_idx, p_idx, match_pd_ind, match_p_idx, smiles, rotate_third_axis=True)
            if elements==None:
                exit("ERROR: elements is None %s"%(smiles))
            if len(elements)==0:
                exit("ERROR: elements is empty %s"%(smiles))

            coords, elements = utils.xtb_opt(coords, elements, smiles)
            coords, elements = utils.xtb_opt(coords, elements, smiles)
            coords, elements = utils.xtb_opt(coords, elements, smiles)

            P_index=elements.index("P")
            As_index=elements.index("As")
            elements[As_index]="N"
            settings["P_index"]=P_index
            if not success:
                exit("ERROR: Pd addition did not work. Exit.")


        elif settings["add_Pd_Cl2"] and "Pd" not in smiles:
            metal_char="Pd"
            settings["metal_char"]=metal_char
            num_bonds_P=utils.get_num_bonds_P(smiles)
            smiles_Hs=utils.add_Hs_to_P(smiles, num_bonds_P)
            #smiles=utils.add_to_smiles(smiles,"[Pd]([Cl])[Cl]")
            spacer_smiles="[Pd]([As+](F)(F)F)([As+](F)(F)F)[As+](F)(F)F"
            smiles_incl_spacer = utils.add_to_smiles(smiles_Hs, spacer_smiles)

            coords_ligand_complex, elements_ligand_complex = utils.get_coords_from_smiles(smiles_incl_spacer, suffix, conversion_method)
            coords_ligand, elements_ligand = utils.remove_complex(coords_ligand_complex, elements_ligand_complex, smiles, settings)
            P_index, bond_indeces = utils.get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)
            direction=np.zeros((3))
            for bond_index in bond_indeces:
                direction+=(coords_ligand[bond_index]-coords_ligand[P_index])
            direction/=(-np.linalg.norm(direction))
            coords_ligand=np.array(coords_ligand.tolist()+[(coords_ligand[P_index]+2.25*direction).tolist()])
            elements_ligand.append("Pd")
            match_pd_ind=len(elements_ligand)-1
            match_p_idx=P_index
            # replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, rotate_third_axis=True)
            coords_pd, elements_pd, pd_idx, p_idx = geo_utils.get_Pd_Cl_Cl()
            success, coords, elements = geo_utils.replace(coords_pd, elements_pd, coords_ligand, elements_ligand, pd_idx, p_idx, match_pd_ind, match_p_idx, smiles, rotate_third_axis=True)
            if elements==None:
                exit("ERROR: elements is None %s"%(smiles))
            if len(elements)==0:
                exit("ERROR: elements is empty %s"%(smiles))
            P_index=elements.index("P")
            settings["P_index"]=P_index
            if not success:
                exit("ERROR: Pd addition did not work. Exit.")

        else:
            coords, elements = utils.get_coords_from_smiles(smiles, suffix, conversion_method)

        xyzfilename1="%s/%s.xyz"%(inputdir, molname)
        xyzfilename2="%s/%s_kraken.xyz"%(inputdir, molname)
        xyzfilenames=[xyzfilename1]#,xyzfilename2]
        for xyzfilename in xyzfilenames:
            if not os.path.exists(xyzfilename):
                utils.exportXYZ(coords, elements, xyzfilename)
                break
    else:
        coords, elements = utils.readXYZ("%s/%s.xyz"%(inputdir, filename))
        if add_Ni_CO_3:
            metal_char="Ni"
            settings["metal_char"]=metal_char
        #smiles=open("smiles_codes/%s.smi"%(molname),"r").readlines()[0].split()[0]


    # prepare some directories
    os.chdir(calcdir)

    #if os.path.exists(moldir):
    #    exit("calculations/%s already exists. please remove it if you want to do the same calculation again."%(moldir))
    utils.try_mkdir(moldir)



    # run crest
    print("   ---   Run crest calculation of molecule %s"%(molname))
    xyzfilename="%s.xyz"%(molname)
    crest_done, xtb_done, coords_all, elements_all, boltzmann_data_conformers, conf_indeces, electronic_properties_conformers, time_needed = utils.run_crest(coords, elements, moldir, xyzfilename, settings, smiles)
    if not crest_done:
        exit("ERROR: something went wrong with crest calculation of %s"%(moldir))
    if not xtb_done:
        exit("ERROR: something went wrong with xtb calculations of %s"%(moldir))
    print("   ---   Found %i conformers of molecule %s"%(len(elements_all),molname))

    # run morfeus
    morfeus_parameters_conformers = []

    for conf_idx, coords_conf in enumerate(coords_all):
        dummy_positions = electronic_properties_conformers[conf_idx]["dummy_positions"]
        elements_conf=elements_all[conf_idx]
        moldir_conf = "%s/conf_%i"%(moldir,conf_indeces[conf_idx])
        print("   ---   Run morfeus calculation of molecule %s, conformer %i out of %i"%(molname,conf_idx+1,len(coords_all)))
        morfeus_parameters = utils.run_morfeus(coords_conf, elements_conf, dummy_positions, moldir_conf, settings, smiles)
        morfeus_parameters_conformers.append(morfeus_parameters)


    # go back to the start directory
    os.chdir(startdir)

    # save data
    print("   ---   Save the results of molecule %s to %s/%s.yml"%(molname,resultsdir, molname))
    if crest_done and xtb_done:
        '''
        print("\n   ###   ELEMENTS")
        print(elements_all)
        print("\n   ###   CONFORMERS")
        for conf_idx,conf in enumerate(coords_all):
            print("conformer %i"%(conf_idx))
            print(conf)
        print("\n   ###   DATA")
        print(boltzmann_data_conformers)
        print("\n   ###   TIME")
        print(time_needed)
        print("\n   ###   DATA CONFORMERS")
        for conf_idx,confdata in enumerate(electronic_properties_conformers):
            print("conformer %i"%(conf_idx))
            print(confdata)
        '''


        # start putting data in the results dictionary
        results_here={}
        results_here["coords_start"]=coords.tolist()
        results_here["elements_start"]=elements
        results_here["smiles"]=smiles

        # calculate number of rotatable bonds
        if smiles=="not available":
            rotatable_bonds=[]
        else:
            try:
                rotatable_bonds=utils.get_rotatable_bonds(smiles)
            except:
                rotatable_bonds=[]
        results_here["rotatable_bonds"]=rotatable_bonds
        num_rotatable_bonds=len(rotatable_bonds)
        results_here["num_rotatable_bonds"]=num_rotatable_bonds

        # add conformation data
        # until this point, all the data is still available
        for conf_idx in range(len(elements_all)):
            coords_conf = coords_all[conf_idx]
            elements_conf = elements_all[conf_idx]
            boltzmann_data_conf = boltzmann_data_conformers[conf_idx]
            electronic_properties_conf = electronic_properties_conformers[conf_idx]
            morfeus_parameters_conf = morfeus_parameters_conformers[conf_idx]
            results_here["conf_%i"%(conf_idx)] = {"coords":coords_conf,
                                                  "elements":elements_conf,
                                                  "boltzmann_data":boltzmann_data_conf,
                                                  "electronic_properties":electronic_properties_conf,
                                                  "sterimol_parameters":sterimol_parameters_conf
                                                  }


        # sort the data in different output files and kill unnecessary data
        data_here, data_here_confs, data_here_esp_points = utils.reduce_data(results_here)
        # add the timings
        time2=time.time()
        time_all=time2-time1
        results_here["settings"]=settings
        results_here["time_crest"]=time_needed[0]
        results_here["time_morfeus"]=time_needed[1]
        results_here["time_all"]=time_all

        # save the main output file (this will hopefully be the smallest file with the most important data
        outfilename="%s/%s/%s.yml"%(startdir,resultsdir, molname)
        outfile=open(outfilename,"w")
        outfile.write(yaml.dump(data_here, default_flow_style=False))
        outfile.close()

        # conformer data goes to an extra output file
        outfilename_confs="%s/%s/%s_confs.yml"%(startdir,resultsdir, molname)
        outfile_confs=open(outfilename_confs,"w")
        outfile_confs.write(yaml.dump(data_here_confs, default_flow_style=False))
        outfile_confs.close()

        utils.combine_csvs(molname, resultsdir, data_here, data_here_confs)


    else:
        print("   ---   molecule %s FAILED"%(molname))
        outfilename="%s/%s/%s.yml"%(startdir,resultsdir, molname)
        outfile=open(outfilename,"w")
        outfile.write("FAILED\n")
        outfile.close()
        

    print("   ###   Finished molecule %s"%(molname))







