# 201006 TGE: add finding computation that is missing a termination in the last job as ERR - this was previously only found if this happened in the first=only job in a log file
import os,shutil
import pathlib as pl
import re
ligname = re.compile("[0-9]{8}")  

def copy_other():
    cwd = pl.Path.cwd()
    files = [i.name for i in cwd.iterdir() if (i.is_file() and ligname.match(i.name) and i.suffix != ".log") ]

    for conformer in os.listdir("."):
        if conformer.split(".")[-1] == "log":
            continue
        print(conformer)

        if conformer.split(".")[-1] in ["cub","txt"]:
            conformer = conformer.split("_Pesp")[0]
            try:
                os.stat(dir+conformer[:8]+"/"+conformer+"/")
                shutil.move(conformer,dir+conformer[:8]+"/"+conformer+"/"+conformer)
            except:
                print("not moving "+conformer)
        if conformer.split(".")[-1] in ["fchk","com","txt","wfn"]:
            conformer = conformer.split(".")[0]     
            for rem in ["_nbo","_sp","_pop","_freq","_g16","_opt","_nmr","_npa","_fukui_results","_f","_out","_Pesp","_in","_vmin_results2","_vmin_results"]:
                conformer = conformer.replace(rem,"")
            try:
                os.stat(dir+conformer[:8]+"/"+conformer+"/")
                shutil.move(conformer,dir+conformer[:8]+"/"+conformer+"/"+conformer)
            except:
                print("not moving "+conformer)
    return()    

def split_log(ligand, conformer):
    cwd = pl.Path.cwd()
    print("Splitting log:", conformer)
    with open(cwd/ligand/conformer/f"{conformer}.log") as f:
        loglines = f.readlines()

    # ligand = ligname.match(conformer)[0]
     
    # jobs = [] # 0: route line. 1: number of start line. 2: number of termination line
    routes   = [] # route line of each subjob
    chsp     = [] # charge/spin of each subjob
    types    = [] # type of subjob
    starts   = [0,] # index of start line 
    ends     = [] # index of end line
    termination = [] # Normal/Error

    for i in range(len(loglines)):
        if " #" in loglines[i] and "---" in loglines[i-1] and "---" in loglines[i+1]:
            routes.append(loglines[i].strip())
        if " #" in loglines[i] and "---" in loglines[i-1] and "---" in loglines[i+2]:
            routes.append(loglines[i].strip() + loglines[i+1].strip())
        if  re.search("Charge = [-+]?[0-9 ]+multiplicity",loglines[i],re.IGNORECASE):
            chsp.append([int(x) for x in re.findall("[-+]?[0-9]+",loglines[i])])
        if "Normal termination" in loglines[i]:
            ends.append(i)
            starts.append(i+1)
            termination.append("Normal")
        if "Error termination" in loglines[i] and "Error termination" not in loglines[i-1]:
            termination.append("Error")
            ends.append(i+3)
    if len(ends) < len(routes):
        ends.append(-2)
        termination.append("none")
    # if len(ends)==0:
    #     ends.append(-2)    

    done = True
    for i,route in enumerate(routes):
        if re.search("opt",route,re.IGNORECASE):
            types.append("opt")
        elif re.search("freq",route,re.IGNORECASE) and not re.search("opt",route,re.IGNORECASE):
            types.append("freq")
        elif re.search("wfn",route,re.IGNORECASE):
            types.append("sp")
        elif re.search("nmr",route,re.IGNORECASE):
            types.append("nmr")
        elif re.search("efg",route,re.IGNORECASE):
            types.append("efg")
        elif re.search("nbo",route,re.IGNORECASE) and chsp[i] == [0,1]:
            types.append("nbo")
        elif re.search("nbo",route,re.IGNORECASE) and chsp[i] == [-1,2]:
            types.append("ra")
        elif re.search("nbo",route,re.IGNORECASE) and chsp[i] == [1,2]:
            types.append("rc")
        elif re.search("scrf",route,re.IGNORECASE):
            types.append("solv")
       
        if len(termination)==0:
            print("no termination found. exit.")
            done = False
        elif termination[i] == "Error":
            print("ERROR in termination found. exit.")
            done = False
        elif termination[i] == "none":
            print("ERROR: last job did not terminate. exit.")
            done = False
            
        if not done:
            if not (cwd/"ERR").exists():
                (cwd/"ERR").mkdir()
            with open(cwd/"ERR"/f"{conformer}_{types[i]}.log","w") as f:
                for line in loglines[starts[i]:ends[i]+1]:
                    f.write(line)
            print(f"               Error in {types[i]}")
            return("Error")
        else:
            # if not (cwd/ligand).exists():
            #     (cwd/ligand).mkdir()
            # if not (cwd/ligand/conformer).exists():
            #     (cwd/ligand/conformer).mkdir()
            with open(cwd/ligand/conformer/f"{conformer}_{types[i]}.log", "w") as f:
                for line in loglines[starts[i]:ends[i]+1]:
                    f.write(line)
    # otherfiles = [i.name for i in cwd.iterdir() if (i.stem==conformer and i.suffix != ".log")]
    # for of in otherfiles:
    #     shutil.move(cwd/of,cwd/ligand/conformer/of)
    return("")

if __name__ == '__main__': 
    cwd = pl.Path.cwd()
    if not (cwd/"ERR").exists():
        (cwd/"ERR").mkdir()

    ligname = re.compile("[0-9]{8}")
    ligands = sorted([i.name for i in cwd.iterdir() if (ligname.match(i.name) and i.is_dir())])
    conformers = {ligand: [i.name for i in (cwd/ligand).iterdir() if i.is_dir()] for ligand in ligands}

    # logs = sorted([i.stem for i in cwd.iterdir() if (ligname.match(i.name) and i.suffix == ".log")])
    # for log in logs:
        # split_log(log[:8],log[:-4])
            
    for ligand in ligands:
        for conformer in conformers[ligand]:
            #logs = [i.name for i in (cwd/ligand/conformer).rglob("*.log")]
            #if f"{conformer}.log" in logs and f"{conformer}_opt.log" not in logs:
            split_log(ligand,conformer)
    
     
