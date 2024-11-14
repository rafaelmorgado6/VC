
import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

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
        cv2.imshow('img',img)
        cv2.waitKey(500)

    return ret, corners



# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Lê todas todas as imagens left.jpg
images = sorted(glob.glob('..//images//left*.jpg'))


for fname in images:    # Percorre fram por frame
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret == True:
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

# Guarda intrinsic e distortion matrices num ficheiro .npz 
np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)

cv2.waitKey(-1)
cv2.destroyAllWindows()