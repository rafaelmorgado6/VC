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
def FindAndDisplayChessboard(img):
    
    # Converte a imagem de colorida (BGR) para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Encontra os cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

     # Se o chessboard for encontrado:
    if ret:
        
        # cv2.cornerSubPix -> Melhora a precisão
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        
        # Desenha os cantos refinados na imagem original
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)
        
        return ret, corners2  # Return refined corners
    
    return ret, corners


# Lê a imagem e vê se dá erro 
img = cv2.imread('/home/rafa/Desktop/VC/images/left02.jpg')  # Load the original image
if img is None:
    print("Error: Image not found.")
    exit()

# Encontra e exibe os cantos do tabuleiro de xadrez na imagem
ret, corners = FindAndDisplayChessboard(img)
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

# Camera calibration
ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)


# Resultados da calibração
print("Intrinsics: ")
print(intrinsics)   
print("Distortion: ")
print(distortion)
for i in range(len(tvecs)):
    print("Translations(%d): " % i)
    print(tvecs[i])
    print("Rotation(%d): " % i)
    print(rvecs[i])



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

# Obtém o primeiro vetor de rotação e vetor de translação
rvec = rvecs[0]
tvec = tvecs[0]

# Projeta os pontos 3D para 2D usando os parâmetros da câmera
projected_points, _ = cv2.projectPoints(cube_points, rvec, tvec, intrinsics, distortion)

# Converte os pontos projetados para um formato apropriado para desenho
projected_points = projected_points.reshape(-1, 2)

# Define as arestas do cubo que serão desenhadas
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Base square
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
]

# Desenha as arestas do cubo na imagem
for edge in edges:
    start_point = tuple(projected_points[edge[0]].astype(int))  # Ponto inicial da aresta
    end_point = tuple(projected_points[edge[1]].astype(int))    # Ponto final da aresta
    cv2.line(img, start_point, end_point, (0, 255, 0), 2)   # Desenha a aresta em verde com espessura 2

# Display the result
cv2.imshow('Projected Cube', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
