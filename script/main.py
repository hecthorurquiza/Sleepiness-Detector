import cv2
import mediapipe as mp
import time
from eyes import coord_left_eye, coord_right_eye, all_eyes, calc_ear
from mounth import coord_mounth, calc_mar


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
is_sleep = 0
blinking = 0


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucess, frame = cap.read()
        if not sucess:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        # Redimensiona o frame do video
        frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)
        
        length, width, _ = frame.shape
        
        frame = cv2.flip(frame, 1) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        escape_facemesh = facemesh.process(frame) # Processamos a imagem com o Face Mesh
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            for face_landmarks in escape_facemesh.multi_face_landmarks:
                # Desenhamos os pontos detectados na face
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                        # Mudamos a cor, espessura e tamanho dos pontos da face
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1), 
                                        # Mudamos a cor, espessura e tamanho das conexões da face
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1))
                
                for id_coord, coord_xyz in enumerate(face_landmarks.landmark):
                    if id_coord in all_eyes:
                        # Desenhamos círculos mais escuros na região dos olhos para destacá-los
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, width, length)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                    
                    if id_coord in coord_mounth:
                        # Destacamos os contornos mais internos da boca
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, width, length)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                
                ear = calc_ear(face_landmarks.landmark, coord_right_eye, coord_left_eye)
                mar = calc_mar(face_landmarks.landmark, coord_mounth)
                
                # Se a boca e os olhos estiverem fechados marcamos o tempo inicial
                if ear < 0.33 and mar <= 0.1:
                    t_initial = time.time() if is_sleep == 0 else t_initial
                    blinking = blinking + 1 if is_sleep == 0 else blinking # Contando o número de piscadas
                    is_sleep = 1
                    
                # Se os olhos abrirem e 'is_sleep' ou permanecerem fechados mas com a boca a aberta resetamos o status de 'is_sleep'
                if (is_sleep == 1 and ear >= 0.31) or (ear <= 0.31 and mar >= 0.1):
                    is_sleep = 0
                
                # Definimos o tempo inicial ao fechar os olhos enquanto o tempo final continua a ser contado pelo loop
                t_final = time.time()
                
                # Subtraimos o tempo final pelo inicial constantemente enquanto os olhos permanecerem fechados     
                t = (t_final-t_initial) if is_sleep == 1 else 0.0
    
                if t >= 1.5: 
                    # Desenhando um retângulo
                    cv2.rectangle(frame, (200, 400), (450, 440), (109, 233, 219), -1)
                    
                    # Texto personalizado
                    cv2.putText(frame, "COM SONOLENCIA", (210, 430), cv2.FONT_HERSHEY_DUPLEX, 0.85, (58,58,55), 1)

        except:
            pass

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == 27:
            break
        
cap.release()
cv2.destroyAllWindows()