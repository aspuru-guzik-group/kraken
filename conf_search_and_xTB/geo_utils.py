import os
import numpy as np
import scipy.spatial as scsp
import scipy.linalg as scli


def get_Ni_CO_3():
    crest_best=""" Ni         -2.05044275300666    0.06382544955011    0.09868120676498
 P          -2.80714796997979   -1.10266971180507   -1.69574169412280
 C          -2.69200378269657   -0.76605024888162    1.57419568293391
 O          -3.04257804499007   -1.20995335174270    2.55963300719774
 C          -2.69223663646763    1.74898458637508   -0.06255834794434
 O          -3.04279673881760    2.82969533618590   -0.06960307962299
 C          -0.24189533762829    0.01881947327896    0.02959721559736
 O           0.89275735454075    0.05117679841698    0.07869727019190"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def get_Pd_NH3_Cl_Cl():
    crest_best=""" Pd         -1.89996002172552   -0.02498444632011    2.10982622577294
 N          -1.56965112209091   -2.05219215877655    2.00001618954387
 As          -2.21595829857879    2.00450177777031    2.22007410905701
 H          -2.40942129799767    2.36875215537164    1.28398287161819
 H          -3.02318569399418    2.18955283434028    2.82011004940424
 H          -1.37353382758245    2.44891664756754    2.59391276210718
 Cl          0.35060095551484    0.32532669157403    2.26937306191342
 Cl         -4.15039897250316   -0.37607926860031    1.97331323844022"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def get_Pd_PH3_Cl_Cl():
    crest_best=""" Pd        -0.0000038844        0.0000159819        0.0000111133
 P         -1.6862635579       -1.4845823545        0.0000219312
 As         1.6863052034        1.4845534610        0.0000263723
 H          1.5596931931        2.8713746717        0.0001941369
 H          2.5992646617        1.3913133533       -1.0337086367
 H          2.5995574579        1.3910615548        1.0334736685
 Cl        -1.8219820508        1.3831400099       -0.0000386628
 Cl         1.8219489915       -1.3831565314       -0.0000388596"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def get_Pd_Cl_Cl():
    crest_best=""" Pd        -0.0000038844        0.0000159819        0.0000111133
 P         -1.6862635579       -1.4845823545        0.0000219312
 Cl        -1.8219820508        1.3831400099       -0.0000386628
 Cl         1.8219489915       -1.3831565314       -0.0000388596"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def rotationMatrix(vector,angle):
    angle=angle/180.0*np.pi
    norm=(vector[0]**2.0+vector[1]**2.0+vector[2]**2.0)**0.5
    direction=vector/norm

    matrix=np.zeros((3,3))
    matrix[0][0]=direction[0]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[1][1]=direction[1]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[2][2]=direction[2]**2.0*(1.0-np.cos(angle))+np.cos(angle)

    matrix[0][1]=direction[0]*direction[1]*(1.0-np.cos(angle))-direction[2]*np.sin(angle)
    matrix[1][0]=direction[0]*direction[1]*(1.0-np.cos(angle))+direction[2]*np.sin(angle)

    matrix[0][2]=direction[0]*direction[2]*(1.0-np.cos(angle))+direction[1]*np.sin(angle)
    matrix[2][0]=direction[0]*direction[2]*(1.0-np.cos(angle))-direction[1]*np.sin(angle)

    matrix[1][2]=direction[1]*direction[2]*(1.0-np.cos(angle))-direction[0]*np.sin(angle)
    matrix[2][1]=direction[1]*direction[2]*(1.0-np.cos(angle))+direction[0]*np.sin(angle)

    return(matrix)



def replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, smiles, rotate_third_axis=True):

    # copy all the initial things to not change the original arrays
    c1=np.copy(c1_i)
    e1=np.copy(e1_i)
    c2=np.copy(c2_i)
    e2=np.copy(e2_i)

    clash_dist=1.0

    # shift the ligand
    c1-=c1[P_index]
    # shift the ferrocene
    c2-=c2[match_P_index]

    # rotate He-P-axis of ligand
    dir1=c1[Au_index]-c1[P_index]
    dir1/=scli.norm(dir1)
    dir2=np.array([0.0,1.0,0.0])
    dir2/=scli.norm(dir2)
    if np.abs(1.0-np.sum(dir1*dir2))>1e-3:
        cross_dir1_dir2=np.cross(dir1,dir2)
        cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
        angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
        rotation=rotationMatrix(cross_dir1_dir2, angle)
        coords_rotated=[]
        for atom in c1:
            coords_rotated.append(np.dot(rotation, atom).tolist())
        c1=np.array(coords_rotated)


    # rotate P-He_replacement-axis of ligand
    dir1=c2[match_Au_index]-c2[match_P_index]
    dir1/=scli.norm(dir1)
    dir2=np.array([0.0,1.0,0.0])
    dir2/=scli.norm(dir2)
    if np.abs(1.0-np.sum(dir1*dir2))>1e-3:
        cross_dir1_dir2=np.cross(dir1,dir2)
        cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
        angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
        rotation=rotationMatrix(cross_dir1_dir2, angle)
        coords_rotated=[]
        for atom in c2:
            coords_rotated.append(np.dot(rotation, atom).tolist())
        c2=np.array(coords_rotated)
    #c2+=np.array([0.0,0.7,0.0])
    # rotatble bonds to P
    #print(smi1)
    #rot_bonds=get_rotatable_bonds(smi1)
    #print(rot_bonds)
    #print(Au_index, P_index)


    if rotate_third_axis:
        # rotate third axis
        axis2=np.array([0.0,1.0,0.0])
        axis2/=scli.norm(axis2)
        #min_dist_opt=1.0
        min_best=clash_dist
        angle2_best=None

        all_steps=[]
        all_elements=[]
        for angle2 in np.linspace(0.0,360.0,361):
            rotation2=rotationMatrix(axis2, angle2)
            # shift to zero
            coords_rotated2=[]
            for atom in c2:
                coords_rotated2.append(np.dot(rotation2, atom))
            coords_rotated2=np.array(coords_rotated2)

            all_steps.append(np.copy(coords_rotated2))
            all_elements.append(e2)

            # shift back
            mask1=np.ones((len(c1)))
            mask1[Au_index]=0
            mask1[P_index]=0
            mask2=np.ones((len(c2)))
            mask2[match_Au_index]=0
            mask2[match_P_index]=0
            indeces1=np.where(mask1==1)[0]
            indeces2=np.where(mask2==1)[0]
            min_dist=np.min(scsp.distance.cdist(c1[indeces1],coords_rotated2[indeces2]))

            if min_dist>min_best: #min_dist>min_dist_opt and 
                min_best=min_dist
                angle2_best=angle2
                #print("found better RMSD: %f"%(RMSD_best))

        if angle2_best == None:
            #print("FAILED")
            print("ERROR: Did not find a good rotation angle without clashes! %s"%(smiles))
            return(False,None,None)


        rotation2=rotationMatrix(axis2, angle2_best)
        # shift to zero
        coords_rotated_final=[]
        for atom in c2:
            coords_rotated_final.append(np.dot(rotation2, atom))
        c2=np.array(coords_rotated_final)


    c_final=[]
    e_final=[]
    c2_final=[]
    e2_final=[]
    for idx in range(len(c1)):
        if idx!=P_index:
            c_final.append(c1[idx].tolist())
            e_final.append(e1[idx])
    for idx in range(len(c2)):
        if idx!=match_Au_index:
            c_final.append(c2[idx].tolist())
            e_final.append(e2[idx])
            c2_final.append(c2[idx].tolist())
            e2_final.append(e2[idx])

    c_final=np.array(c_final)

    #all_steps.append(np.copy(c2_final))
    #all_elements.append(["K" for e in e2_final])

    #all_steps.append(np.copy(c_final))
    #all_elements.append(e_final)


    #exportXYZs(all_steps,all_elements,"group_rotation.xyz")

    e_final=[str(x) for x in e_final]
    return(True, c_final, e_final)


