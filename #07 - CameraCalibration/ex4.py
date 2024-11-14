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



# Função de calibração externa para uma nova imagem
def ExternalCalibration(img, objp):
    ret, corners = FindAndDisplayChessboard(img)
    if ret:
        # Calibração externa usando solvePnP
        success, rvec, tvec = cv2.solvePnP(objp, corners, intrinsics, distortion)
        if success:
            print("Vetor de rotação (rvec):\n", rvec)
            print("Vetor de translação (tvec):\n", tvec)
            return rvec, tvec
    return None, None




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

#capture.release()
cv2.destroyAllWindows()

  # Guarda intrinsic e distortion matrices num ficheiro .npz 
np.savez('camera_params.npz', intrinsics=intrinsics, distortion=distortion)

# Carrega os parâmetros salvos
with np.load('camera_params.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']
print("Parâmetros carregados:")
print("Intrínsecos:\n", intrinsics)
print("Distorção:\n", distortion)


# Exemplo de uso: carrega uma nova imagem e realiza a calibração externa
ret, new_img = capture.read()
if not ret:
        print("Erro ao capturar imagem.")
        

if ret:
    rvec, tvec = ExternalCalibration(new_img, objp)
 
    if rvec is not None:

        projected_points, _ = cv2.projectPoints(cube_points, rvec, tvec, intrinsics, distortion)
        projected_points = projected_points.reshape(-1, 2)
        
        for edge in edges:
            start_point = tuple(projected_points[edge[0]].astype(int))
            end_point = tuple(projected_points[edge[1]].astype(int))
            cv2.line(new_img, start_point, end_point, (0, 255, 0), 2)
        
        
        # Verifica se há imagens capturadas para exibir
        if captured_images:
            print("Exibindo todas as imagens capturadas com o cubo projetado...")
            for img in captured_images:
                img = cv2.flip(img, 1)
                cv2.imshow("Captured Images with Projected Cube", img)
                cv2.waitKey(1000)  # Mostra cada imagem por 1 segundo (1000 ms)
        else:
            print("Nenhuma imagem foi capturada.")

# Libera os recursos e fecha todas as janelas
capture.release()
cv2.destroyAllWindows()

