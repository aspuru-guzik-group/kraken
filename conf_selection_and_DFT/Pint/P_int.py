import sys,subprocess,re,os,shutil
import numpy as np
import pathlib as pl
cwd = pl.Path.cwd()

import ded # from Robert Pollice. Minor modification to accept a working directory as argument
import read_geom # TG

float_pattern = re.compile(r"[-+]?\d*\.\d+")
Pintresults = ["Pint_P_int","Pint_dP","Pint_P_min","Pint_P_max","volume","surface_area","sphericity"]


command = 'Multiwfn'


def run_Multiwfn(wd,name,ext):
    inputargs = "12\n2\n-2\n3\n0.25\n0\n7\nq\n"
    multiwfn = subprocess.run(f"{command} {name}{ext}",stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="ascii",shell=True,input=inputargs,cwd=wd)
    # a = subprocess.Popen("Multiwfn " + compfile, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,cwd=wd,shell=True)
    # subprocess.run("mv", cwd / "vtx.txt", cwd / dir+filename+"_vdw.txt",cwd=wd)
    with open(wd/(f"{name}_Multiwfn_out.txt"),"w") as f:
        f.write(multiwfn.stdout)
    with open(wd/(f"{name}_Multiwfn_err.txt"),"w") as f:
        f.write(multiwfn.stderr)
    os.rename(wd/("vtx.txt"),wd/(f"{name}_vtx.txt"))
    return()

def run_Multiwfn_win(wd,name,ext):
    if ext in [".fch",".fchk"]:
        inputargs = f"12\n2\n-2\n3\n0.25\n0\n7\n-1\n-1\n100\n2\n5\n{name}.wfn\nq\n"
    else:
        inputargs = "12\n2\n-2\n3\n0.25\n0\n7\nq\n"
    multiwfn = subprocess.run(f"{command} {name}{ext}",stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="ascii",shell=True,input=inputargs,cwd=wd)
    # a = subprocess.Popen("Multiwfn " + compfile, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,cwd=wd,shell=True)
    # subprocess.run("mv", cwd / "vtx.txt", cwd / dir+filename+"_vdw.txt",cwd=wd)
    with open(wd/(f"{name}_Multiwfn_out.txt"),"w") as f:
        f.write(multiwfn.stdout)
    with open(wd/(f"{name}_Multiwfn_err.txt"),"w") as f:
        f.write(multiwfn.stderr)
    os.rename(wd/("vtx.txt"),wd/(f"{name}_vtx.txt"))
    return()

def run_Multiwfn_win_promol(wd,name):
    inputargs = "5\n-1\n1\n3\n2\n0\n12\n2\n-2\n1\n1\n0.00174\n3\n0.25\n0\n7\nq\n"
    multiwfn = subprocess.run(f"{command} {name}.xyz",stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="ascii",shell=True,input=inputargs,cwd=wd)
    with open(wd/(f"{name}_Multiwfn_out.txt"),"w") as f:
        f.write(multiwfn.stdout)
    with open(wd/(f"{name}_Multiwfn_err.txt"),"w") as f:
        f.write(multiwfn.stderr)
    os.rename(wd/("vtx.txt"),wd/(f"{name}_vtx.txt"))
    os.rename(wd/("density.cub"),wd/(f"{name}_density.cub"))
    return()

def read_disp(wd,name,disp):
    with open(wd/(f"{name}_{disp}.out"),"r") as f:
        disp_cont = f.readlines()
    
    if disp == "d4":
        P_pat = "15 P"
        start_pat = "#   Z        covCN         q      C6AA      C8AA      α(0)"
    elif disp == "d3":
        P_pat = " p "
        start_pat = "XYZ [au]"
    for ind,line in enumerate(disp_cont[::-1]):
        if start_pat in line:
            for line_ in disp_cont[:-ind-1:-1]:
                if P_pat in line_:
                    if disp == "d4":
                        dispres = [float(i) for i in line_.split()[3:]]
                    elif disp == "d3":
                        dispres = [None,None]+[float(i) for i in line_.split()[-3:-1]]+[None]
                    return(dispres)
    return([None,None,None,None,None])

def read_multiwfnout(wd,name):
    try:
        with open(wd/(f"{name}_Multiwfn_out.txt"),"r") as f:
            mwfn_cont = f.readlines()
        mwfn_pat = "================= Summary of surface analysis ================="
        for ind,line in enumerate(mwfn_cont[::-1]):
            if mwfn_pat in line:
                vol = float(float_pattern.findall(mwfn_cont[-ind+1])[-1])  # in Angstrom^3
                area = float(float_pattern.findall(mwfn_cont[-ind+4])[-1]) # in Angstrom^2
                sph = np.round(np.pi**(1/3)*(6*vol)**(2/3)/area,6) # Sphericity
                return([vol,area,sph])
    except:
        return([None,None,None])
    
def read_dedout(wd,name,disp):
    with open(wd/(f"{name}_ded_{disp}.txt"),"r") as f:
        ded_cont = f.readlines()
    ded_results = [float(ded_cont[i].split()[-1]) for i in range(2,6)]
    return(ded_results)


def P_int_main(name="",directory="./",disp = "d3",promol=False):
    # option promol: if no orbital information is found, generate P_int from promolecular density
    # wd = cwd/directory
    wd = pl.Path(directory)

    foundextensions = [i.suffix for i in wd.iterdir() if i.stem == name]

    try:
        with open(wd/(f"{name}_ded_{disp}_summary.txt"),"r") as f:
            results = [float(i) for i in f.readlines()[1].split(";")]
            # return(results[:7])
            return({Pintresults[i]:results[i] for i in range(7)})
    except:
        pass
    
    try:
        os.stat(wd/(f"{name}_vtx.txt")) # surface data
        os.stat(wd/(f"{name}_Multiwfn_out.txt")) # surface data
    except:
        if promol == True:
            if ".xyz" not in foundextensions:
                raise ValueError("No suitable input file found. Needs a .xyz file. Exiting.")
            try:
                os.stat(wd/"atomwfn")
            except:
                os.mkdir(wd/"atomwfn")
                atomwfn_orig = pl.Path("/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/Pint/Multiwfn_3.7_bin_Linux_noGUI/examples/atomwfn")
                [shutil.copyfile(atomwfn_orig/aof,wd/"atomwfn"/aof) for aof in os.listdir(atomwfn_orig)]
            run_Multiwfn_win_promol(wd,name)

        else:
            # locate orbital information
            found = False
            for extension in [".wfn",".fchk",".fch"]:
                if os.path.isfile(wd/(name+extension)):
                    found = True
                    # run Multiwfn to generate vtx.txt
                    run_Multiwfn_win(wd,name,extension)
                    break
            if found == False:
                raise ValueError("No suitable input file found. Needs a .wfn file or formatted checkfile. Exiting.")

    try:
        os.stat(wd/(f"{name}.xyz"))
    except:
        coords = read_geom.read(wd,name+extension)
        read_geom.write_xyz(coords,name,wd)
        
    # run ded.py
    try:
        os.stat(wd/(f"{name}_ded_{disp}.txt"))
    except:
        ded_path = '/uufs/chpc.utah.edu/common/home/u1209999/PL_workflow/new_org_use/Pint'
        t = subprocess.run(f"python {ded_path}/ded.py ./{name}.xyz ./{name}_vtx.txt --charge 0 --disp {disp}",cwd=wd,shell=True)
        
    # read results
    results = read_dedout(wd,name,disp)+read_multiwfnout(wd,name)+read_disp(wd,name,disp)
    with open(wd/(f"{name}_ded_{disp}_summary.txt"),"w") as f:
        f.write("Pint;dP;Pmin;Pmax;Volume/A^3;Area/A^2;Sphericity;covCN(P);q(P);C6AA(P);C8AA(P);alpha(0)(P)\n")
        f.write(";".join([str(i) for i in results])) 

    try:
        os.stat(cwd/"ded_results.txt")
    except:
        with open(cwd/"ded_results.txt","w") as f:
            f.write("Name;Pint;dP;Pmin;Pmax;Volume/A^3;Area/A^2;Sphericity;covCN(P);q(P);C6AA(P);C8AA(P);alpha(0)(P)\n")
    with open(cwd/"ded_results.txt","a") as f:
        f.write(";".join([name]+[str(i) for i in results])+"\n") 
    # return(results[:7])
    return({Pintresults[i]:results[i] for i in range(7)})

if __name__ == "__main__":
    try:
        os.stat("ded_results.txt")
    except:
        with open("ded_results.txt","w") as f:
            f.write("Name;Pint;dP;Pmin;Pmax;Volume/A^3;Area/A^2;Sphericity;covCN(P);q(P);C6AA(P);C8AA(P);alpha(0)(P)\n")
    files = [file for file in os.listdir(cwd) if 'Pd' not in file]
    for file in files: 
        if file.split(".")[-1] == "fchk":
        	P_int_main(name=file.split(".")[-2])
        elif file.split(".")[-1] == "xyz":
            P_int_main(name=file.split(".")[-2],promol=True)
        
