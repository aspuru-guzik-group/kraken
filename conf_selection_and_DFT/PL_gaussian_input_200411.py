


def get_link0(n):
    mem = int(round(96/40*n-0.5))  #int(round(96/40*n-0.5))
    return("%%nprocs=%i\n%%mem=%iGB\n"%(n, mem))

# jobsetup: keywords for individual jobs and respective spin/charge card
jobsetup = { 
"sp_1": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ density=current output=wfn ","0 1",],
"sp_nmr": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ nmr=giao ","0 1",],
"sp_efg": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ prop=efg ","0 1",],
"sp_nbo": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ pop=nbo7 ","0 1",],
"sp_solv": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ scrf=(SMD,solvent=chloroform) ","0 1",],
"sp_rc": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ pop=nbo7 ","1 2",],
"sp_ra": ["# PBE1PBE/def2TZVP int=(grid=ultrafine) empiricaldispersion=GD3BJ pop=nbo7 ","-1 2",],
"opt_freq": ["# PBEPBE/6-31+G(d,p) denfit int=(grid=ultrafine) empiricaldispersion=GD3BJ opt freq=noraman ","0 1",],
"opt_recalc": ["# PBEPBE/6-31+G(d,p) denfit int=(grid=ultrafine) empiricaldispersion=GD3BJ opt=(recalcfc=50) freq=noraman ","0 1",],
}

# add options and a list of keys from 'jobsetup' to get other job combinations
joboptions = {
    # "all":["opt_freq","sp_1","sp_nmr","sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],
    "all":["opt_recalc","sp_1","sp_nmr","sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],

    "re_from_sp":["sp_1","sp_nmr","sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],
    "re_from_nmr":["sp_nmr","sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],
    "re_from_efg":["sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],
    "re_from_nbo":["sp_nbo","sp_solv","sp_rc","sp_ra"],
    "re_from_solv":["sp_solv","sp_rc","sp_ra"],
    "re_from_rc":["sp_rc","sp_ra"],
    "re_from_ra":["sp_ra"],

    "sp_only":["sp_1","sp_nmr","sp_efg","sp_nbo","sp_solv","sp_rc","sp_ra"],
    "sp_os":["sp_rc","sp_ra"],
    "opt":["opt_recalc"]
    }

# these ECPs will be used if a heavy atom is present, as defined in 'heavy'
ecps = {
    "sp":"SDD",     # for single points
    "op":"Lanl2DZ"  # for optimization/frequency
}


## 
import os,sys,re

#for basis/ECP selection. Br has been removed
heavy = ["K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"] #,"Br"
        
basisset_pat = re.compile("/\S*")

def metaldetector(geom,job):
    elements = [i.strip().split(" ")[0].title() for i in geom.strip().split("\n")]
    heavyelems = " ".join(set([x for x in elements if x in heavy]))
    otherelems = " ".join(set([x for x in elements if x not in heavy]))
    if len(heavyelems) != 0:
        isheavy = True
        basisset = re.search(basisset_pat,jobsetup[job][0])[0]
        basiscard = f"{otherelems} 0\n{basisset[1:]}\n****\n{heavyelems} 0\n{ecps[job[:2]]}\n****\n\n{heavyelems} 0\n{ecps[job[:2]]}\n\n"
        if re.search("6-31\+*G",basisset):
            newroute = re.sub(basisset_pat," gen 6D pseudo=read",jobsetup[job][0])
        else:
            newroute = re.sub(basisset," gen pseudo=read",jobsetup[job][0])
    else:
        isheavy = False
        basiscard = ""
        newroute = jobsetup[job][0]
    return(isheavy,newroute,basiscard)

def get_geom_xyz(filename,ext):
    with open(f"{filename}.{ext}","r") as f:
        n_atoms = int(f.readline().strip())
        cont = "".join(f.readlines()[1:n_atoms+1])
    return(cont)

def get_geom_gin(filename, ext):
    with open(f"{filename}.{ext}","r") as f:
        filecont = f.readlines()
    start = 999
    for l in range(len(filecont)):
        if len(filecont[l].split()) == 2 and len(filecont[l+1].split()) == 4: # charge/spin line
            start = l+1
        if len(filecont[l].split()) < 2 and l > start:
            end = l
            break
    return("".join(filecont[start:end]))

def get_geom_gout(filename,ext): # gets the compressed stream information at the end of a Gaussian job
    streams = []
    starts,ends = [],[]
    error = "failed or incomplete job" # default unless "normal termination" is in file

    with open(f"{filename}.{ext}") as f:
        loglines = f.readlines()

    for i in range(len(loglines)):
        if "1\\1\\" in loglines[i]:
            starts.append(i)
        if "@" in loglines[i]:
            ends.append(i)
        if "Normal termination" in loglines[i]:
            error = ""
    if len(starts) != len(ends) or len(starts) == 0: #probably redundant
        error = "failed or incomplete job"
        print(f"{filename}.{ext}\n{error}")
        return("")

    for i in range(len(starts)):
        tmp = ""
        for j in range(starts[i],ends[i]+1,1):
            tmp = tmp + loglines[j][1:-1]
        #print(tmp)
        streams.append(tmp.split("\\"))

    geom = []
    for item in streams[-1][16:]:
        if item == "":
            break
        geom.append("  ".join([item.split(",")[0],item.split(",")[-3],item.split(",")[-2],item.split(",")[-1]]))
    return("\n".join(geom)+"\n")

get_geoms = {"xyz":get_geom_xyz,
             "com":get_geom_gin,
             "gjf":get_geom_gin,
             "log":get_geom_gout,
             "LOG":get_geom_gout,
             "out":get_geom_gout,         
         }

def write_coms(directory, name, suffix, geometry, joboption, num_processors=40):
    filecontent = ""
    read = 0 # write "geom=check guess=read"
    link = 0 # write "--Link1--" at the beginning
    writegeom = 1 # write the geometry
    for job in joboptions[joboption]:
        has_heavymetal,route,basiscard = metaldetector(geometry,job)
        if "output=wfn" in route: 
            wfnline = f"{name}.wfn\n\n"
        else:
            wfnline = ""
        if (jobsetup[job][1] != "0 1" and joboption != "sp_os") or "scrf" in route:
            chkline = f"%oldchk={name}.chk\n%chk={name}_{job}.chk\n"
        else: 
            chkline = f"%chk={name}.chk\n"

        filecontent += link*"--Link1--\n" + get_link0(num_processors) + chkline + route + read*"geom=check guess=read "+f"\n\n{name}_{job}\n\n" + jobsetup[job][1]+"\n"+geometry.title()*writegeom + "\n" + basiscard + wfnline

        if joboption != "sp_os":
            read = 1
            writegeom = 0
        link = 1
    filecontent += "\n"

    with open(f"{directory}/{name}{suffix}.com","w", newline='\n') as f:
        f.write(filecontent)

if __name__ == '__main__': 
    directory = "./" 
    if len(sys.argv)>1:
        ext = sys.argv[1]
        joboption = sys.argv[2]
    else:
        ext = "xyz"
        joboption = "all"
    files = [f.split(".")[0] for f in os.listdir(directory) if f.endswith((ext))]
    geometries = {f:get_geoms[ext](f"{directory}/{f}",ext) for f in files}
    
    for k,v in geometries.items():
        write_coms(directory,k,v,joboption)


