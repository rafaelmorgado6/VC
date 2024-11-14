import numpy as np
import cv2


# Função para encontrar e exibir o tabuleiro de xadrez
def FindAndDisplayChessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)

        return ret, corners2  # Return refined corners
    
    return ret, corners

# Board Size
board_h = 9
board_w = 6

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(board_w-1, board_h-1, 0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane.
captured_images = []  # List to store images with the projected cube

# Captura de vídeo da câmera
capture = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
images_count = 0


print("Move o tabuleiro de xadrez para a câmera. Pressione 'c' para capturar uma imagem.")



while images_count < 3:
    ret, img = capture.read()  # Lê a imagem da câmera
    if not ret:
        print("Erro ao capturar imagem.")
        break

    img = cv2.flip(img, 1)     
    cv2.imshow('Camera', img)  # Mostra a imagem capturada

    key = cv2.waitKey(1)
    if key == ord('c'):  # Se 'c' for pressionado, captura a imagem
        ret, corners = FindAndDisplayChessboard(img)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Calibração da câmera
            ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
            
            # Define os pontos 3D de um cubo
            cube_points = np.array([
                [0, 0, 0],    # Ponto 0
                [1, 0, 0],    # Ponto 1
                [1, 1, 0],    # Ponto 2
                [0, 1, 0],    # Ponto 3
                [0, 0, -1],   # Ponto 4
                [1, 0, -1],   # Ponto 5
                [1, 1, -1],   # Ponto 6
                [0, 1, -1],   # Ponto 7
            ], dtype=np.float32)

            # Obtém o primeiro vetor de rotação e vetor de translação
            rvec = rvecs[-1]  # Usando o último par de vetores
            tvec = tvecs[-1]

            # Projeta os pontos 3D para 2D usando os parâmetros da câmera
            projected_points, _ = cv2.projectPoints(cube_points, rvec, tvec, intrinsics, distortion)

            # Converte os pontos projetados para um formato apropriado para desenho
            projected_points = projected_points.reshape(-1, 2)

            # Desenha as arestas do cubo na imagem
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Base square
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
            for edge in edges:
                start_point = tuple(projected_points[edge[0]].astype(int))  # Ponto inicial da aresta
                end_point = tuple(projected_points[edge[1]].astype(int))    # Ponto final da aresta
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)   # Desenha a aresta em verde com espessura 2

            captured_images.append(img)  # Armazena a imagem com o cubo projetado
            images_count += 1  # Incrementa o contador de imagens capturadas
            print(f"Imagem {images_count} capturada com o cubo projetado.")

    if key == ord('q'):  # Pressione 'q' para sair
        break

capture.release()
cv2.destroyAllWindows()

# Exibe todas as imagens capturadas em um loop infinito
while True:  
    for img in captured_images:
        cv2.imshow('Captured Images with Cube', img)  # Mostra a imagem com o cubo projetado
        if cv2.waitKey(800) & 0xFF == ord('q'):  # Se 'q' for pressionado, sai do loop
            break  # Sai do loop interno se 'q' for pressionado
    else:
        continue  # Continua exibindo as imagens se não sair
    break  # Sai do loop externo se 'q' foi pressionado
  
  # Guarda intrinsic e distortion matrices num ficheiro .npz 
np.savez('7.3_calibration_parameters.npz', intrinsics=intrinsics, distortion=distortion)

cv2.destroyAllWindows()
