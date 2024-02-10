import numpy as np

coord_left_eye = [385, 380, 387, 373, 362, 263]
coord_right_eye = [160, 144, 158, 153, 33, 133]
all_eyes = coord_left_eye + coord_right_eye

def calc_ear(landmark, coord_right_eye, coord_left_eye):
    try:
        landmark = np.array([[coord.x, coord.y] for coord in landmark])
        left_eye = landmark[coord_left_eye,:]
        right_eye = landmark[coord_right_eye, :]

        left_ear = (np.linalg.norm(left_eye[0]-left_eye[1])+np.linalg.norm(left_eye[2]-left_eye[3]))/(2*(np.linalg.norm(left_eye[4]-left_eye[5])))
        right_ear = (np.linalg.norm(right_eye[0]-right_eye[1])+np.linalg.norm(right_eye[2]-right_eye[3]))/(2*(np.linalg.norm(right_eye[4]-right_eye[5])))
    
    except:
        left_ear = 0.0
        right_ear = 0.0
        
    mean_ear = (left_ear + right_ear) / 2
    return mean_ear