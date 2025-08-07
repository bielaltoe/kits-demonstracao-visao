import cv2
import mediapipe as mp
# import serial

video = cv2.VideoCapture(0)
# arduino = serial.Serial('COM14', 9600)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
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
    pontos = []

    if handPoints:
        for points in handPoints:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))
            
            # IDs dos dedos (ponta dos dedos)
            dedos = [8, 12, 16, 20]  # indicador, meio, anelar, mindinho
            contador = 0

            # Verificar polegar (ID 4)
            # Para mão direita: polegar esquerdo da ponta está levantado
            # Para mão esquerda: polegar direito da ponta está levantado
            if len(pontos) > 4:
                if pontos[4][0] > pontos[3][0]:  # Polegar para direita
                    contador += 1

            # Verificar outros dedos
            for x in dedos:
                if len(pontos) > x and pontos[x][1] < pontos[x-2][1]:
                    contador += 1

            cv2.putText(img, str(contador), (100, 100), font, 4, (255, 0, 0), 5)
            # Quando Arduino estiver conectado, descomente:
            # arduino.write(str(contador).encode() + b'\n')

    cv2.imshow("hands", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
# arduino.close()  # Descomente quando usar Arduino