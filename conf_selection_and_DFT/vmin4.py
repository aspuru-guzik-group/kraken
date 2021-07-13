import os,sys,re
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
from shutil import copyfileobj
import bz2
from distutils.util import strtobool

# parameters
BA = 0.529177 # Bohr - Angstrom conversion
numbers_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")    

# box dimensions:
# standard presets for Phosphines
class Grid_def:
    def __init__(self,size):
        if size == 0:
            self.dxy = 0.85/BA   # dimensions perpendicular to P-LP
            self.dz = 0.8/BA   #0.5 dimension in P-LP direction
            self.d_lp = 2.0/BA #1.9# distance of grid center from P
            self.npoints = [25,25,40]     # number of grid points in x,y,z directions
        elif size == 1:
            self.dxy = 1.9/BA   # dimensions perpendicular to P-LP
            self.dz = 1.0/BA   #0.5 dimension in P-LP direction
            self.d_lp = 2.15/BA #1.9# distance of grid center from P
            self.npoints = [50,50,50]     # number of grid points in x,y,z directions


            
dxy = 0.85/BA   # dimensions perpendicular to P-LP
dz = 0.8/BA   #0.5 dimension in P-LP direction
d_lp = 2.0/BA #1.9# distance of grid center from P
npoints = [25,25,40]     # number of grid points in x,y,z directions

# alternative presets for very electron-poor phosphines
# dxy = 1.5/BA   # dimensions perpendicular to P-LP
# dz = 1.50/BA    # dimension in P-LP direction
# d_lp = 2.5/BA  # distance of grid center from P
# npoints = [30,30,50]     # number of grid points in x,y,z directions
# dxy = 1.9/BA   # dimensions perpendicular to P-LP
# dz = 1.0/BA    # dimension in P-LP direction
# d_lp = 2.0/BA  # distance of grid center from P
# npoints = [50,50,50]     # number of grid points in x,y,z directions

# for N ligands
# dxy = 0.9/BA   # dimensions perpendicular to P-LP
# dz = 0.5/BA    # dimension in P-LP direction
# d_lp = 1.4/BA  # distance of grid center from P
# npoints = [25,25,25]     # number of grid points in x,y,z directions

rcov = {
"H": 0.32,"He": 0.46,"Li": 1.2,"Be": 0.94,"B": 0.77,"C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,"Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,"Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,"Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,"Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,"Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,"Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.08,"Ag": 1.15,"Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,"Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,"Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,"Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,"Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,"Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,"Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,"Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55
}

elements = {
"1": "H",
"5": "B",
"6": "C",
"7": "N",
"8": "O",
"9": "F",
"14": "Si",
"15": "P",
"16": "S",
"17": "Cl",
"26": "Fe",
"33": "As",
"34": "Se",
"35": "Br",
"44": "Ru",
"46": "Pd",
"51": "Sb",
"53": "I",
}

class Vminob:
    def __init__(self,name,ext,status):
        self.name = name
        self.ext = ext
        if status == "" or status == "stdcube":
            self.dxy = 0.85/BA   # dimensions perpendicular to P-LP
            self.dz = 0.8/BA   #0.5 dimension in P-LP direction
            self.d_lp = 2.0/BA #1.9# distance of grid center from P
            self.npoints = [25,25,40]     # number of grid points in x,y,z directions
        elif status == "follow_edge":
            self.dxy = 1.9/BA   # dimensions perpendicular to P-LP
            self.dz = 1.0/BA   #0.5 dimension in P-LP direction
            self.d_lp = 2.15/BA #1.9# distance of grid center from P
            self.npoints = [50,50,50]     # number of grid points in x,y,z directions

    def p_lp(self,iteration):       
        # expects a three-coordinate P containing compound. Adds a valence to P so that the angle to the three previous substituents is maximized and resorts the coordinate output for convenience

        for i,atom in enumerate(self.coords):
            if (atom[0] == "P" or atom[0] == "As" or atom[0] == "Sb") and sum(self.conmat[i]) <=3:
                self.nop = i
                self.coordp = np.array(self.coords[i][1:])
                break

        vec = np.array([0.0,0.0,0.0])
        bonded = []
        for atom in range(len(self.coords)):
            if self.conmat[self.nop][atom]:
                bonded.append(atom)
                coorda = np.array(self.coords[atom][1:])
                vec += self.coordp - coorda
        self.coordlp = self.d_lp*vec/np.linalg.norm(vec) + self.coordp  # coordinates of grid center

        atomno = max(iteration-1,0)
        dir_bond1 = np.array((self.coords[bonded[atomno]][1:]))-np.array((self.coords[self.nop][1:])) # direction of first P-R bond    
        dirx = np.cross(dir_bond1,vec)  # defines the x-direction of the grid
        diry = np.cross(vec,dirx)       # defines the y-direction of the grid

        self.dirx = dirx/np.linalg.norm(dirx)    # normalization of the grid-coordinate system
        self.diry = diry/np.linalg.norm(diry)
        self.dirz = vec/np.linalg.norm(vec)
        self.grid_coords = [self.dirx*self.dxy/self.npoints[0],self.diry*self.dxy/self.npoints[1],self.dirz*self.dz/self.npoints[2]]
    #    grid_coords = [2*dirx/npoints[0],2*diry/npoints[1],dirz/2*npoints[2]]
        self.grid_or = self.coordlp - self.dz*0.5 * self.dirz - 0.5*self.dxy * self.diry - 0.5*self.dxy * self.dirx # grid_origin
        return()

    def copy_info(self,prev):
        self.coords = prev.coords
        self.conmat = prev.conmat
        self.nop = prev.nop
        self.coordp = prev.coordp
        self.coordlp = prev.coordlp
        self.dirx = prev.dirx
        self.diry = prev.diry
        self.dirz = prev.dirz
        self.grid_coords = prev.grid_coords
        self.grid_or = prev.grid_or

    def follow_vmin(self,coords_vmin_prev):
        vec_P_vmin_prev = coords_vmin_prev - self.coordp
        grid_center = self.coordp + vec_P_vmin_prev / np.linalg.norm(vec_P_vmin_prev) * (np.linalg.norm(vec_P_vmin_prev)+0.5)
        self.dirx = np.cross(self.coordlp,vec_P_vmin_prev)
        self.diry = np.cross(vec_P_vmin_prev,self.dirx)
        self.dirx = self.dirx/np.linalg.norm(self.dirx)    # normalization of the grid-coordinate system
        self.diry = self.diry/np.linalg.norm(self.diry)
        self.dirz = vec_P_vmin_prev/np.linalg.norm(vec_P_vmin_prev)
        self.grid_coords = [self.dirx*self.dxy/self.npoints[0],self.diry*self.dxy/self.npoints[1],self.dirz*self.dz/self.npoints[2]]
        self.grid_or = grid_center - self.dz*0.5 * self.dirz - 0.5*self.dxy * self.diry - 0.5*self.dxy * self.dirx # grid_origin

        # print("coords_vmin_prev: {:.2f},{:.2f},{:.2f}\ngrid_center: {:.2f},{:.2f},{:.2f}\ngrid_or: {:.2f},{:.2f},{:.2f}\n".format(*coords_vmin_prev,*grid_center,*self.grid_or))

    def read_cubtxt(self,directory="./"):
        name = directory + self.name
        get_permission(name + "_Pesp_out"+self.suffix+".txt")
        with open(name + "_Pesp_out"+self.suffix+".txt") as f:
            cubtxt = f.readlines()
        self.esp = np.array(([float(line.split()[-1]) for line in cubtxt]))
        self.v_min = np.amin(self.esp)
        self.vmin_ind = int(np.where(self.esp==self.v_min)[0][0])
        self.coords_min = np.array(([float(i) for i in cubtxt[self.vmin_ind].split()[:3]]))  # find index of v_min, get coordinates in Bohr
        return()

    def analyze_vmin(self,directory="./"):
        npoints = self.npoints
        name = directory + self.name
        self.r_min = np.linalg.norm(self.coords_min - self.coordp)*BA
        self.line_pos = self.vmin_ind + 1

        if self.line_pos % npoints[2] in [0,1]:
            self.on_edge = True
        elif (self.line_pos <= npoints[2] * npoints[1]) or (npoints[2] * npoints[1] * npoints[0] - self.line_pos <= npoints[2] * npoints[1]):
            self.on_edge = True
        elif (self.line_pos % (npoints[2] * npoints[1]) <= npoints[2]) or (npoints[2] * npoints[1] - (self.line_pos % (npoints[2] * npoints[1])) <= npoints[2]):
            self.on_edge = True
        else:
            self.on_edge = False

        self.vminatom,self.vmindist = "",""
        rmin_other = {(i[0]+str(j+1)): np.linalg.norm(self.coords_min - np.array(i[1:])) * BA for j,i in enumerate(self.coords) if i[0] not in  ["H","P"]}
        rmin_other_S = pd.Series(rmin_other)

        if rmin_other_S.min() < self.r_min/1.1: # scaled P to account for larger radius vs. most other elements
            self.wrongatom = True
            # print("{};Vmin of other atom;{}".format(self.name,rmin_other_S.idxmin()))
            self.vminatom = rmin_other_S.idxmin()
            self.vmindist = rmin_other_S.min()
        else:
            self.wrongatom = False

        self.coords_min_A = self.coords_min * BA

        with open(name + "_vmin_results2.txt","a") as f:
            f.write("{};{};{};{:.5f};{};{};{:.4f};{:.4f};{:.4f}\n".format(self.name,self.suffix,self.v_min,self.r_min,self.on_edge,self.wrongatom,*self.coords_min_A))
        
        # print(self.name,";",self.r_min)
        
        return()

    def get_geom_cub(self,directory="./"):
        # reads the geometry out of a cube file (in Bohr)
        file = directory + self.name + "_Pesp_out.cub" 
        self.coords = []
        self.grid_coords = [[],[],[]]
        get_permission(file)
        with open(file) as f:
            fcont = f.readlines()
        natoms = int(re.findall(numbers_pattern,fcont[2])[0])
        self.grid_or = np.asarray([float(i) for i in re.findall(numbers_pattern,fcont[2])[1:]])
        for i in range(3):
            self.npoints[i] = int(re.findall(numbers_pattern,fcont[3+i])[0])
            self.grid_coords[i] = np.asarray([float(j) for j in re.findall(numbers_pattern,fcont[3+i])[1:]])
        self.dirx = self.grid_coords[0]/np.linalg.norm(self.grid_coords[0])    # normalization of the grid-coordinate system
        self.diry = self.grid_coords[1]/np.linalg.norm(self.grid_coords[1])
        self.dirz = self.grid_coords[2]/np.linalg.norm(self.grid_coords[2])

    # self.dxy = self.grid_coords[0]*self.npoints[0]/self.dirx
    # self.dz = self.grid_coords[2]*self.npoints[2]/self.dirz
    # self.d_lp = (self.grid_or - (self.coordp - self.dz*0.5 * self.dirz - 0.5*self.dxy * self.diry - 0.5*self.dxy * self.dirx))/self.dirz

    


        for line in fcont[6:6+natoms]:
            linesplit = re.findall(numbers_pattern,line)
            if len(linesplit) != 5:
                break
            if linesplit[0] not in elements.keys():
                print("Element not implemented in code: "+linesplit[0])
                continue
            else:  
                self.coords.append([elements[linesplit[0]]]+[float(i) for i in linesplit[2:]])
        return()


def get_permission(filepath):
    if not os.access(filepath,os.R_OK):
        os.chmod(filepath, 0o777)


def get_files(ext):
    files = []
    for file in os.listdir("."):
        if file[-len(ext):] == ext:
            files.append(file)
    return(files)

def get_geom_fch(vminob,directory="./"):
    file = directory + vminob.name + "." + vminob.ext
    get_permission(file)
    with open(file) as f:
        natoms = 250
        content = []
        for line in f:
            content.append(line)
            if len(content) == 3:
                natoms = int(content[-1].split()[-1])
            elif "Atomic numbers" in line:
                atomnumbersstart = len(content)-1
            elif "Nuclear charges" in line:
                atomnumbersend = len(content)-1
            elif "Current cartesian coordinates" in line:
                coordsstart = len(content)-1
            elif "Number of symbols" in line:
                coordsend = len(content)-1
                break
            elif "Force Field" in line:
                coordsend = len(content)-1
                break

    atoms,atomnos = [],[]
    for line in range(atomnumbersstart+1,atomnumbersend):
        atomnos += content[line].split()
    for atom in atomnos:
        if atom in elements:
            atoms.append(elements[atom]),
        else: 
            atoms.append(atom)

    coords = []
    coordsstream = []
    for line in range(coordsstart+1,coordsend):
        coordsstream += content[line].split()
    coordsstream = [float(i) for i in coordsstream]
    for atom in range(natoms):
        coords.append([atoms[atom]]+coordsstream[atom*3:atom*3+3])
    #print(coords)
    return(coords)

def get_conmat(rcov, coords): # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    natom = len(coords)
    max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    for i in range(0,natom):
        if coords[i][0] not in rcov.keys():
            continue
        for iat in range(0,natom):
            if coords[iat][0] not in rcov.keys():
                continue
            if iat != i:
                dx = coords[iat][1] - coords[i][1]
                dy = coords[iat][2] - coords[i][2]
                dz = coords[iat][3] - coords[i][3]
                r = np.linalg.norm([dx,dy,dz])*BA # with conversion Bohr - Angstrom
                rco = rcov[coords[i][0]]+rcov[coords[iat][0]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0+np.math.exp(-k1*(rr-1.0)))
                if damp > 0.85: 
                    conmat[i,iat],conmat[iat,i] = 1,1
    return(conmat)

def write_inpcube(vminob,directory="./"):
    name = directory + vminob.name
    writecont = " " + name + " potential=scf\n Electrostatic potential from Total SCF Density\n"
    writecont += "{0:>5}{1[0]:>12.6f}{1[1]:>12.6f}{1[2]:>12.6f}\n".format(len(vminob.coords),vminob.grid_or)
    for i in range(3):
        writecont += "{0:>5}{1[0]:>12.6f}{1[1]:>12.6f}{1[2]:>12.6f}\n".format(vminob.npoints[i],vminob.grid_coords[i])

    with open(name+"_Pesp_in"+vminob.suffix+".cub","w",newline="\n") as f:
        f.write(writecont)   
    return()

def run_cubman(n_in,suffix,directory="./"):
    name = directory + n_in
    a = subprocess.Popen("cubman", stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    inputargs = "to\n"+ name + "_Pesp_out"+suffix+".cub\ny\n" + name + "_Pesp_out"+suffix+".txt\n"
    a_out = a.communicate(input=inputargs.encode())    
    return()

def run_cubegen(n_in,suffix,directory="./"):
    name = directory + n_in
    #nproc = str(multiprocessing.cpu_count() - 2)
    nproc = '1'
    try: 
        os.stat(name + "_Pesp_out"+suffix+".cub")
    except:
        a = subprocess.run("cubegen "+nproc+" potential=scf " + name + ".fchk " + name + "_Pesp_out"+suffix+".cub -1 h " + name + "_Pesp_in"+suffix+".cub",shell=True, stdout=subprocess.PIPE)
    os.remove(name + "_Pesp_in"+suffix+".cub")
    return()

def read_cubes(n_in,suffix,directory="./"):
    name = directory + n_in
    get_permission(name+"_Pesp_out"+suffix+".cub")
    with open(name+"_Pesp_out"+suffix+".cub") as f:
        cube = f.readlines()
    natoms = int(cube[2].split()[0])
    esp = np.array(([float(num) for line in cube[6+natoms:] for num in line.split()]))
    with open(directory+"vmin_results.txt","a") as f:
        f.write("%s;%s\n" %(n_in,np.amin(esp)))
    return(np.amin(esp))

# def analyze_vmin(self,writefile=True,suffix=""):
#     self.rmin = np.linalg.norm(self.coords_vmin - self.coordp) * BA
#     vec0 = self.coords_vmin - self.coordlp


#     line_pos = self.vmin_ind + 1
#     if line_pos % self.npoints[2] in [0,1]:
#         self.on_edge = "true"
#     elif (line_pos <= self.npoints[2] * self.npoints[1]) or (self.npoints[2] * self.npoints[1] * self.npoints[0] - line_pos <= self.npoints[2] * self.npoints[1]):
#         self.on_edge = "true"
#     elif (line_pos % (self.npoints[2] * self.npoints[1]) <= self.npoints[2]) or (self.npoints[2] * self.npoints[1] - (line_pos % (self.npoints[2] * self.npoints[1])) <= self.npoints[2]):
#         self.on_edge = "true"
#     else:
#         self.on_edge = "false"

#     #on_edge = "false"
#     #for i in range(3):
#     #    crit = np.linalg.norm(self.grid_info[i+1] * (self.npoints[i] - 2) / 2.0)    # should be (self.npoints[i] - 1) instead of 2
#     #    dist = np.absolute(np.dot(vec0, norm_vec(self.grid_info[i + 1])))
#     #    print(dim[i],self.grid_info[i+1],self.npoints[i],crit,dist)
#     #    if dist > crit:
#     #        on_edge = "true"
#     #        break
#     self.vminatom,self.vmindist = "",""
#     rmin_other = {(i[0]+str(j+1)): np.linalg.norm(self.coords_vmin - np.array(i[1:])) * BA for j,i in enumerate(self.coords) if i[0] not in  ["H","P"]}
#     rmin_other_S = pd.Series(rmin_other)

#     if rmin_other_S.min() < self.rmin/1.1: # scaled P to account for larger radius vs. most other elements
#         self.wrongatom = True
#         print("{};Vmin of other atom;{}".format(self.name,rmin_other_S.idxmin()))
#         self.vminatom = rmin_other_S.idxmin()
#         self.vmindist = rmin_other_S.min()
#     else:
#         self.wrongatom=False
#     coords_min_A = self.coords_vmin * BA
#     if writefile:
#         with open("vmin_results.txt", "a") as f:
#             f.write("%s;%s;%s;%s;%s;coordinates;%s;%s;%s;%s;%s\n" % (self.name,suffix, self.vmin, self.rmin, self.on_edge, coords_min_A[0], coords_min_A[1], coords_min_A[2],self.vminatom,self.vmindist))
#     return()


def get_vmin(infile,directory="./",runcub=False):
    if "_Pesp_out" in infile:
        name =  infile.split("_Pesp_out.")[0]
        ext = infile.split("_Pesp_out.")[1]
    else:
        name = infile.split(".")[0]
        ext = infile.split(".")[1]  

    vminobjects = []

    # if f"{name}_vmin_results2.txt" in os.listdir(directory):
        # with open(directory+name+"_vmin_results2.txt","r") as f:
            # vmincont = f.readlines()
        # for line in vmincont[1:]:     
            # result = vmincont[-1].strip().split(";")
            # vminob = Vminob(name,"txt","done")
            # vminob.suffix = result[1]
            # vminob.v_min = float(result[2])
            # vminob.r_min = float(result[3])
            # vminob.on_edge = bool(strtobool(result[4]))
            # vminob.wrongatom = bool(strtobool(result[4]))
            # vminobjects.append(vminob)
            # if not vminob.on_edge and not vminob.wrongatom:
                # return(vminob)  
        # stdcubes = [i for i in vminobjects if "stdcube" in i.suffix]
        # if len(stdcubes) == 3:
            # mincubeno = np.argmin([i.v_min for i in stdcubes])
            # return(stdcubes[mincubeno])        
 
        # try:
        #     os.stat(name+"_vmin_results.txt")
        #     print("reading results "+ name)
        #     v_min,r_min = read_vmin_results(name,"./")
        #     with open("vmin_results.txt","a") as f:
        #         f.write("%s;%s\n" %(v_min,r_min))
        # except:
            #print("writing cub " + name)

    with open(directory+name+"_vmin_results2.txt","w") as f:
        f.write("Name;Suffix;Vmin;R(Vmin);On_edge;Wrong_atom;X_Vmin;Y_Vmin;Z_Vmin\n")

    status = ""
    # status:
    #       "": normal/first pass
    #       "follow_edge": previous vmin was on edge of cube
    #       "stdcube": no vmin was found at P; generate three cubes around expected P lone pair
    #       "done": finished procedure

    iteration = 0
    while status != "done":
        vminob = Vminob(name,ext,status)
        if iteration == 0 and "fch" in ext:
            vminob.coords = get_geom_fch(vminob,directory)
            vminob.conmat = get_conmat(rcov, vminob.coords)
        elif iteration == 0 and ext == "cub":
            vminob.get_geom_cub(directory)
            vminob.conmat = get_conmat(rcov, vminob.coords)        
        else:
            vminob.coords = vminobjects[0].coords
            vminob.conmat = vminobjects[0].conmat

        if status == "follow_edge":
            vminob.copy_info(vminobjects[0])
            vminob.follow_vmin(vminobjects[-1].coords_min)
        else:
            vminob.p_lp(iteration)

        vminob.suffix = ("_"+status+str(iteration))*bool(len(status))

        try:
            os.stat(directory+name + "_Pesp_out"+vminob.suffix+".cub")
        except:
            if vminob.ext == "cub":
                if runcub == False: # perform analysis only
                    with open(directory+name + "_vmin_results2.txt","a") as f:
                        f.write("{};{};Problem encountered, run additional cubegens\n".format(vminob.name,vminob.suffix))
                    print("{};{};Problem encountered, run additional cubegens".format(vminob.name,vminob.suffix))
                    break
                elif name+".fchk" in os.listdir(directory):
                    vminob.ext = "fchk"
                    write_inpcube(vminob,directory)
                    run_cubegen(name,vminob.suffix,directory)
                elif name+".fchk.bz2" in os.listdir(directory):
                    try:
                        get_permission(directory+name+".fchk.bz2")
                        with bz2.BZ2File(directory+name+".fchk.bz2", 'rb', compresslevel=9) as i:
                            with open(directory+name+".fchk", 'wb') as o:
                                copyfileobj(i, o)
                        vminob.ext = "fchk"
                        write_inpcube(vminob)
                        run_cubegen(name,vminob.suffix,directory)
                    except:
                        with open(directory+name + "_vmin_results2.txt","a") as f:
                            f.write("{};{};Problem extracting .fchk.bz2 for further analysis\n".format(vminob.name,vminob.suffix))
                        print("{};{};Problem extracting .fchk.bz2 for further analysis\n".format(vminob.name,vminob.suffix))
                        break
                else:
                    with open(directory+name + "_vmin_results2.txt","a") as f:
                        f.write("{};{};Missing .fchk for further analysis\n".format(vminob.name,vminob.suffix))
                    print("{};{};Missing .fchk for further analysis\n".format(vminob.name,vminob.suffix))
                    break
            else:
                write_inpcube(vminob,directory)
                run_cubegen(name,vminob.suffix,directory)

        try:
            os.stat(directory+name + "_Pesp_out"+vminob.suffix+".txt")
        except:
            run_cubman(name,vminob.suffix,directory)

        vminob.read_cubtxt(directory)
        vminob.analyze_vmin(directory)

        vminobjects.append(vminob)

        # print("\non_edge: {}\nwrongatom: {}\niteration: {}\nstatus: {}".format(vminob.on_edge,vminob.wrongatom,iteration,status))
        if vminob.on_edge == False and vminob.wrongatom == False:
            status = "done"
            with open("vmin_results.txt","a") as f:
                f.write("%s;%s;%s\n" %(name,vminob.v_min,vminob.r_min))
            return(vminob)

        elif vminob.on_edge == True and iteration <3 and status != "stdcube":
            status = "follow_edge"
            # print("changed: "+status)
        else: 
            if status != "stdcube":
                iteration = 0
                status = "stdcube"
                # print("changed: "+status)
                
            elif iteration == 3:
                status = "done"
                foundmin = np.argwhere(np.array([not(i.on_edge or i.wrongatom) for i in vminobjects[-3:]]) == True ) # check if any of the three stdcubes found an actual Vmin
                if len(foundmin) > 0:
                    mincubeno = np.argmin([vminobjects[-3+i].v_min for i in foundmin])
                else:
                    mincubeno = np.argmin([i.v_min for i in vminobjects[-3:]])
                mincube = vminobjects[-3+mincubeno]
                with open(directory+name + "_vmin_results2.txt","a") as f:
                    f.write("{};{};{};{:.5f};{};{};{:.4f};{:.4f};{:.4f}\n".format(mincube.name,mincube.suffix,mincube.v_min,mincube.r_min,mincube.on_edge,mincube.wrongatom,*mincube.coords_min_A))

                with open("vmin_results.txt","a") as f:
                    f.write("%s;%s;%s\n" %(mincube.name,mincube.v_min,mincube.r_min))
                return(mincube)

        iteration +=1

    return(vminobjects[-1])

if __name__ == "__main__":
    # arguments can be an individual file of the following kind: name.fch, name.fchk, name.cub
    # if no argument is passed, all .fchk files in the current directory will be parsed. 
    # if no .fchks are present, all .cub files will be parsed.
    todo = []
    if "fch" in sys.argv[-1] or "cub" in sys.argv[-1]:
        todo = [sys.argv[-1]]

    else:
        todo = get_files("fch") + get_files("fchk")
    if len(todo) == 0:
        todo = get_files("_Pesp_out.cub")
    try:
        os.stat("vmin_results.txt")
    except:
        with open("vmin_results.txt","w",newline="\n") as f:
            f.write("\n")   

    for todofile in sorted(todo):
        get_vmin(todofile,"./",True)



         
