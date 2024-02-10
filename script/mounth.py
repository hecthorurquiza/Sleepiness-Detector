import numpy as np

coord_mounth = [82, 87, 13, 14, 312, 317, 78, 308]

def calc_mar(landmark, coord_mounth):
    try:
        landmark = np.array([[coord.x, coord.y] for coord in landmark])
        mounth = landmark[coord_mounth,:]
        mar = (np.linalg.norm(mounth[0]-mounth[1])+np.linalg.norm(mounth[2]-mounth[3])+np.linalg.norm(mounth[4]-mounth[5]))/(2*(np.linalg.norm(mounth[6]-mounth[7])))
    
    except:
        mar = 0.0

    return mar