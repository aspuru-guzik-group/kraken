#!/usr/bin/env python
import os
import sys
import datetime
import shutil
sys.path.append("../../")
import pathlib as pl
from contextlib import suppress

# other py files
import PL_dft_library_201027 as PL_dft_library
import pint_main


# starts at ligand-directory-level
os.system("rm */*.chk")
os.system("rm log_file_analysis")
os.system("for i in $(find */*.log);do echo $i >> log_file_analysis;grep \"Normal termination\" $i | wc -l >> log_file_analysis;done;")

startdir = os.getcwd() # startdir is the ligand level path
ligand_name = startdir.split("/")[-1]

os.chdir("../") # change to top level
cwd = pl.Path.cwd() # cwd is the top level path

logsuffixes = ["opt","freq","efg","nbo","nmr","ra","rc","solv","sp"]

confnames_all = sorted([i.name for i in (cwd/ligand_name).iterdir() if i.is_dir() and ligand_name in i.name])
conformer_errors = {c:[] for c in confnames_all}


# 1. Backup/remove all files from previous parsing (yml,txt)
if not (cwd/ligand_name/"backup_reparsing").exists():
    (cwd/ligand_name/"backup_reparsing").mkdir()  
ymls = [i for i in os.listdir(cwd/ligand_name) if i.endswith(".yml")]
for i in ymls:
    with suppress(FileNotFoundError):
        os.remove(cwd/ligand_name/f"backup_reparsing/{i}")
    shutil.move(cwd/ligand_name/i,cwd/ligand_name/f"backup_reparsing/{i}")

# Delete existing ligand error files
with suppress(FileNotFoundError):
    os.remove(cwd/ligand_name/f"{ligand_name}_errors.txt")

for conformer in confnames_all:
    if not (cwd/ligand_name/conformer/"backup_reparsing").exists():
        (cwd/ligand_name/conformer/"backup_reparsing").mkdir()  
    tobackup = [i for i in os.listdir(cwd/ligand_name/conformer) if i.endswith(".yml")] + [i for i in os.listdir(cwd/ligand_name/conformer) if i.endswith(".txt")]
    for i in tobackup:
        with suppress(FileNotFoundError):
            os.remove(cwd/ligand_name/conformer/f"backup_reparsing/{i}")
        shutil.move(cwd/ligand_name/conformer/i,cwd/ligand_name/conformer/f"backup_reparsing/{i}")    

    # 2. check for combined as well as individual log files to account for the possibility of previously incomplete Gaussian computations that we fixed in the meantime
    try: 
        os.stat(cwd/ligand_name/conformer/f"{conformer}.log")
        # good
        mainlogpresent = True
    except FileNotFoundError:
        # could be bad
        mainlogpresent = False

    sublogs = [i.name for i in (cwd/ligand_name/conformer).glob("*.log") if i.name.split("_")[-1][:-4] in logsuffixes]
    
    if mainlogpresent and len(sublogs) == 9: # combined file there and all subjob files there = ok
        # good
        pass
    elif not mainlogpresent and len(sublogs) == 9: # combined file not there but all subjob files there = ok
        # good
        pass
    elif mainlogpresent and len(sublogs) == 0: # only combined file there = ok if it contains all 9 subjobs
        status = PL_dft_library.split_log(ligand_name, conformer)
        sublogs_new = [i.name for i in (cwd/ligand_name/conformer).glob("*.log") if i.name.split("_")[-1][:-4] in logsuffixes]
        if status != "Error" and len(sublogs_new) == 9:
            # good
            pass
        else:
            # bad!
            sublogs_missing = sorted([i for i in logsuffixes if f"{conformer}_{i}.log" not in sublogs_new])
            err = f"Missing {len(sublogs_missing)} Gaussian log files: {','.join(sublogs_missing)}"
            conformer_errors[conformer].append(err)
            print(f"{ligand_name};{conformer};{err}")
            with open(cwd/ligand_name/f"{ligand_name}_errors.txt","a") as f:
                f.write(f"{ligand_name};{conformer};{err}\n")

    elif mainlogpresent and len(sublogs) != 9: # combined file there, only some of the subjob files there = ok if combined file contains all/all missing subjobs. Split main log to see if it contains the other subjobs
        if not (cwd/ligand_name/conformer/"backup_reparsing").exists():
            (cwd/ligand_name/conformer/"backup_reparsing").mkdir()    
        for i in sublogs:
            try:
                os.rename(cwd/ligand_name/conformer/i,cwd/ligand_name/conformer/f"backup_reparsing/{i}")
            except FileExistsError:  # dunno if that can be a problem
                suf = datetime.datetime.today().strftime("%y%m%d_%H%M")
                os.rename(cwd/ligand_name/conformer/i,cwd/ligand_name/conformer/f"backup_reparsing/{i[:-4]}_{suf}.log")
                
        status = PL_dft_library.split_log(ligand_name, conformer)
        sublogs_new = [i.name for i in (cwd/ligand_name/conformer).glob("*.log") if i.name.split("_")[-1][:-4] in logsuffixes]
        if status != "Error" and len(sublogs_new) == 9: # combined log file contained all sub jobs, won't need the old logfiles any more
            # good
            pass
        elif status != "Error" and len(set(sublogs_new)&set(sublogs)) == 9: # combined log file contained the missing sub jobs, will need the old logfiles, keep the subjobs from the complete log file, if there is overlap
            for i in set(sublogs)-set(sublogs_new):
                os.rename(cwd/ligand_name/conformer/f"backup_reparsing/{i}",cwd/ligand_name/conformer/i)
            # good
            pass
        else:
            # bad!
            sublogs_missing = sorted([i for i in logsuffixes if f"{conformer}_{i}.log" not in set(sublogs_new)&set(sublogs)])
            err = f"Missing {len(sublogs_missing)} Gaussian log files: {','.join(sublogs_missing)}"
            conformer_errors[conformer].append(err)
            print(f"{ligand_name};{conformer};{err}")
            with open(cwd/ligand_name/f"{ligand_name}_errors.txt","a") as f:
                f.write(f"{ligand_name};{conformer};{err}\n")
    elif not mainlogpresent and len(sublogs) != 9: # combined file not there and not all subjob files there = not ok, missing DFT data
        # bad!
        sublogs_missing = sorted([i for i in logsuffixes if f"{conformer}_{i}.log" not in sublogs])
        err = f"Missing main log file and {len(sublogs_missing)} Gaussian log files: {','.join(sublogs_missing)}"
        conformer_errors[conformer].append(err)
        print(f"{ligand_name};{conformer};{err}")
        with open(cwd/ligand_name/f"{ligand_name}_errors.txt","a") as f:
            f.write(f"{ligand_name};{conformer};{err}\n")

    # 3. check for presence of fchk file
    try: 
        os.stat(cwd/ligand_name/conformer/f"{conformer}.fchk")
        # good
    except FileNotFoundError:
        # bad!
        err = f"Missing fchk file."
        conformer_errors[conformer].append(err)
        print(f"{ligand_name};{conformer};{err}")
        with open(cwd/ligand_name/f"{ligand_name}_errors.txt","a") as f:
            f.write(f"{ligand_name};{conformer};{err}\n")

confnames = sorted([k for k,v in conformer_errors.items() if len(v) == 0]) # the conformers with all files present

os.chdir(startdir) # change back to ligand level directory. 
cwd = pl.Path.cwd() # change cwd to the ligand level.

# Begin parsing properties 
print(f"MLD_LOGGER, fname=end.py, ligand={ligand_name}: Pre Pint.")
pint_main.compute_pint(ligand_name) # check if all paths in that file are correct 
print(f"MLD_LOGGER, fname=end.py, ligand={ligand_name}: Post Pint, pre DFT_lib.")
liganddata,status = PL_dft_library.read_ligand(cwd, ligand_name, confnames) # won't need liganddata. status has errors + warnings
print(f"MLD_LOGGER, fname=end.py, ligand={ligand_name}: Post DFT_lib.")
# status = errors,warnings is handled within PL_dft_library.read_ligand() by writing into cwd/ligand_name/f"{ligand_name}_errors.txt"

# Move results
# Todo: write errors,warnings
#       remove backed-up files?
if not os.path.exists("../../dft_results"):
    os.makedirs("../../dft_results")

if os.path.exists("%s_confdata.yml"%(ligand_name)):
    os.system("cp %s_confdata.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_data.yml"%(ligand_name)):
    os.system("cp %s_data.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_relative_energies.csv"%(ligand_name)):
    os.system("cp %s_relative_energies.csv ../../dft_results/."%(ligand_name))

if os.path.exists("%s_errors.txt"%(ligand_name)):
    os.system("cp %s_errors.txt ../../dft_results/."%(ligand_name))

os.chdir("../")
os.system("zip -rq %s %s"%(ligand_name, ligand_name))
if os.path.exists("%s.zip"%(ligand_name)):
    os.system("rm -rf %s"%(ligand_name))


