import numpy as np
import cv2



def getAruCOMarker(image):
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Obtém um dicionário predefinido de marcadores ArUco (padrão de 6x6 celulas e 250 identificadores)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Instancia um conjunto de parâmetros para o detector ArUco, com valores padrão.
    parameters = cv2.aruco.DetectorParameters() 

    # Cria um objeto detector ArUco usando o dicionário de marcadores e os parâmetros definidos
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detecta os marcadores
    corners, ids, rejected = detector.detectMarkers(gray)
    
    print("Marcadores detectados: \n", ids)
    
    # Desenha os contornos dos marcadores na imagem original.
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    return ids is not None, image, corners




# Define os pontos do quadrado para o cálculo de posição/orientação
square_points = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
], dtype=np.float32)

# Carrega os parâmetros da câmera
with np.load('camera.npz') as data:
    intrinsics = data['intrinsics'] # Matriz de parâmetros intrínsecos, que define as propriedades da câmera.
    distortion = data['distortion'] # Coeficientes de distorção da lente da câmera.

# Captura de vídeo
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Cria a janela antes do loop principal
cv2.namedWindow('Janela da Câmera', cv2.WINDOW_AUTOSIZE)

while True:
    # Lê a imagem da câmera
    ret, img = capture.read()
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    # Chama getAruCOMarker para detectar marcadores ArUco no frame capturado
    marker_found, img, corners = getAruCOMarker(img)
    
    # if marker_found:
    #     try:
    #         # Converte os pontos dos cantos para um array de pontos de imagem (2D)
    #         image_points = np.array(corners[0], dtype=np.float32).reshape(-1, 2)

    #         # Calcula a posição e orientação do marcador
    #         retval, rvec, tvec = cv2.solvePnP(square_points, image_points, intrinsics, distortion)

    #         print(f"Rvec: {rvec}")
    #         print(f"Tvec: {tvec}")

    #     except Exception as ex:
    #         print("Erro ao calcular posição/orientação:", ex)


    # Exibe a imagem atualizada na janela existente
    cv2.imshow('Janela da Câmera', img)
    
    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
capture.release()
cv2.destroyAllWindows()
