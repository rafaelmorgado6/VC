import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(board_w-1, board_h-1, 0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


'''Função para encontrar e exibir o tabuleiro de xadrez'''
def  FindAndDisplayChessboard(img):
    
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta os cantos do tabuleiro de xadrez, ret-> indica se o padrão foi encontrado na imagem
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    # Se o padrao chessboard for encontrado, desenha os cantos na imagem e exibe
    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)

    return ret, corners


capture = cv2.VideoCapture(0)   # Captura vídeo da câmera    

if not capture.isOpened():
    print("Erro ao abrir a câmera")
    exit()

window_name = 'window'
cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)



while True:
    ret, img = capture.read()  # Obtém uma imagem inicial da câmera
    #img = cv2.resize(img, (640, 480))  # Reduz a resolução para 640x480

    if not ret:
        print("Erro ao capturar a imagem")
        break

    # Encontra e exibe os cantos do tabuleiro de xadrez na imagem
    ret, corners = FindAndDisplayChessboard(img)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        ret_calib, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

        if ret_calib:
             
            # Obtém o último vetor de rotação e vetor de translação
            rvec = rvecs[-1]  # Usando o último par de vetores
            tvec = tvecs[-1]



            # Define 3D points of a wireframe cube
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

           
            # Projeta os pontos 3D para 2D usando os parâmetros da câmera
            projected_points, _ = cv2.projectPoints(cube_points, rvec, tvec, intrinsics, distortion)

            # Converte os pontos projetados para um formato apropriado para desenho
            projected_points = projected_points.reshape(-1, 2)

            # Define as arestas do cubo que serão desenhadas
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  
                (4, 5), (5, 6), (6, 7), (7, 4),  
                (0, 4), (1, 5), (2, 6), (3, 7)   
            ]

            # Desenha as arestas do cubo na imagem
            for edge in edges:
                start_point = tuple(projected_points[edge[0]].astype(int))  # Ponto inicial da aresta
                end_point = tuple(projected_points[edge[1]].astype(int))    # Ponto final da aresta
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)   # Desenha a aresta em verde com espessura 2
    else:
        # Se o tabuleiro não for encontrado, você pode opcionalmente limpar a lista de objpoints e imgpoints
        objpoints.clear()
        imgpoints.clear()
        

    img = cv2.flip(img, 1)
    cv2.imshow(window_name, img)

    # Se a tecla 'q' for pressionada, sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
capture.release()
cv2.destroyAllWindows()
