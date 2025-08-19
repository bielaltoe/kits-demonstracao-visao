import cv2
import mediapipe as mp
# import serial

video = cv2.VideoCapture(0)
# arduino = serial.Serial('COM14', 9600)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2)  # Alterado para detectar até 2 mãos
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    if not check:
        break
        
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handPoints = results.multi_hand_landmarks
    h, w, d = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    total_dedos = 0  # Contador total para ambas as mãos

    if handPoints:
        for idx, points in enumerate(handPoints):
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            
            # Reinicia pontos para cada mão
            pontos = []
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))
            
            # IDs dos dedos (ponta dos dedos)
            dedos = [8, 12, 16, 20]  # indicador, meio, anelar, mindinho
            contador = 0

            # Identificar se é mão esquerda ou direita
            hand_type = "Right"
            if results.multi_handedness:
                hand_type = results.multi_handedness[idx].classification[0].label
            
            # Corrigindo: o que a câmera vê como "Left" é na verdade a mão direita do usuário e vice-versa
            # Verificar polegar (ID 4) com lógica específica para cada mão
            if len(pontos) > 4:
                # A lógica foi invertida aqui para corresponder à perspectiva do usuário
                if (hand_type == "Left" and pontos[4][0] > pontos[3][0]) or \
                   (hand_type == "Right" and pontos[4][0] < pontos[3][0]):
                    contador += 1

            # Verificar outros dedos
            for x in dedos:
                if len(pontos) > x and pontos[x][1] < pontos[x-2][1]:
                    contador += 1
            
            # Adiciona ao contador total
            total_dedos += contador
            
            # Mostrar contador individual da mão com a correção de perspectiva
            wrist_x, wrist_y = pontos[0]
            user_hand_type = "Esquerda" if hand_type == "Right" else "Direita"
            cv2.putText(img, f"{user_hand_type}: {contador}", (wrist_x - 30, wrist_y - 30), 
                        font, 1, (0, 255, 0), 2)

        # Mostrar contador total
        cv2.putText(img, str(total_dedos), (100, 100), font, 4, (255, 0, 0), 5)
        # Quando Arduino estiver conectado, descomente:
        # arduino.write(str(total_dedos).encode() + b'\n')

    cv2.imshow("hands", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
# arduino.close()  # Descomente quando usar Arduino