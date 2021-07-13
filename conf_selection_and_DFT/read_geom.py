import numpy as np
import re,os
import pathlib as pl

cwd = pl.Path.cwd()
BA = 0.52917721067 # Bohr - Angstrom conversion
float_pattern = re.compile(r"[-+]?\d*\.\d+")

periodictable = ["Bq","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub","Uut","Uuq","Uup","Uuh","Uus","Uuo","X"]

def write_xyz(coord,filename,dir=cwd):     
    a = len(coord)
    out = open(dir/(f"{filename}.xyz"),"w")
    out.write('%s\n\n' %(a))
    for i in range(a):
        #out.write('%2s %'   'f %f %f\n' %(xyz[i][0],xyz[i][1],xyz[i][2],xyz[i][3]))
        out.write("{:s}  {:.7f}  {:.7f}  {:.7f}\n".format(coord[i][0],coord[i][1],coord[i][2],coord[i][3]))
        # out.write('%s %f %f %f\n' %(coord[i][0],coord[i][1],coord[i][2],coord[i][3]))
    out.write('\n')
    out.close
    return  

def div_r(num,den): # divide two integers and round up the result
    return(int(num // den + (num % den > 0)))

def fchk_geom(path,filename):
    with open(path/filename,"r") as f:
        natoms = int([next(f) for x in range(3)][-1].split()[-1])
        nlines = 25+div_r(natoms,6)+div_r(natoms,5)+div_r(natoms,5/3)
        cont = [next(f) for x in range(nlines)]

    for ind,line in enumerate(cont):
        if "Atomic numbers" in line:
            start = ind
            break
    atomnums = np.array("".join(cont[start+1:start+1+div_r(natoms,6)]).split(),dtype=int)
    coords = np.array("".join(cont[start+3+div_r(natoms,6)+div_r(natoms,5):start+3+div_r(natoms,6)+div_r(natoms,5)+div_r(natoms,5/3)]).split(),dtype=np.float).reshape(natoms,3) * BA # in Angstrom 
    geom = []
    for i in range(natoms):
        geom.append([periodictable[atomnums[i]]]+list(np.round(coords[i],8)))
    return(geom)

def wfn_geom(path,filename):
    with open(path/filename,"r") as f:
        natoms = int([next(f) for x in range(2)][-1].split()[-2])
        cont = [next(f) for x in range(natoms)]
    geom = []
    for i in range(natoms):
        geom.append([cont[i].split()[0]]+[round(float(i)*BA,8) for i in float_pattern.findall(cont[i])[:-1]])
    return(geom)

types = {
    "fchk":fchk_geom,
    "fch":fchk_geom,
    "wfn":wfn_geom,
}

def read(path,filename):
    geom = types[filename.split(".")[-1]](path,filename)
    return(geom)

def convert_xyz(path,filename):
    geom = types[filename.split(".")[-1]](path,filename)
    write_xyz(geom,filename.split(".")[0],path)    

if __name__ == "__main__":
    conv = [i for i in os.listdir() if i.split(".")[-1] in types.keys()]    
    for filename in conv: 
        convert_xyz(cwd,filename)
    