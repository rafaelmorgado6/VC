# Importações necessárias
import numpy as np
import cv2
import glob

# Parâmetros do tabuleiro de xadrez
board_h = 9
board_w = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparando pontos 3D no espaço real
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays para armazenar pontos 3D do mundo real e pontos 2D das imagens
objpoints = []      # Pontos 3D
left_corners = []   # Pontos 2D na imagem da câmera esquerda
right_corners = []  # Pontos 2D na imagem da câmera direita

# Função para encontrar cantos da placa de xadrez
def find_corners(img, board_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

# Lendo imagens da câmera esquerda e direita
left_images = sorted(glob.glob('..//images//left*.jpg'))
right_images = sorted(glob.glob('..//images//right*.jpg'))

# Checando se o número de imagens está igual para as duas câmeras
if len(left_images) != len(right_images):
    raise ValueError("O número de imagens para a esquerda e direita não é igual")

# Processamento de cada par de imagens
for left_fname, right_fname in zip(left_images, right_images):
    left_img = cv2.imread(left_fname)
    right_img = cv2.imread(right_fname)

    # Detectando cantos em ambas as imagens
    ret_left, corners_left = find_corners(left_img, (board_w, board_h))
    ret_right, corners_right = find_corners(right_img, (board_w, board_h))

    # Armazenando pontos de calibração se ambos os cantos forem encontrados
    if ret_left and ret_right:
        objpoints.append(objp)
        left_corners.append(corners_left)
        right_corners.append(corners_right)

# Calibração individual de cada câmera
ret1, intrinsics1, distortion1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, left_corners, left_img.shape[1::-1], None, None)
ret2, intrinsics2, distortion2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, right_corners, right_img.shape[1::-1], None, None)

# Calibração estéreo com as estimativas intrínsecas
flags = cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_FIX_INTRINSIC
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    left_corners,
    right_corners,
    intrinsics1,
    distortion1,
    intrinsics2,
    distortion2,
    left_img.shape[1::-1],
    criteria=criteria,
    flags=flags
)

# Salvando as matrizes de calibração para reutilização
np.savez("stereoParams.npz",
         intrinsics1=intrinsics1,
         distortion1=distortion1,
         intrinsics2=intrinsics2,
         distortion2=distortion2,
         R=R, T=T, E=E, F=F)

print("Calibração estéreo concluída e parâmetros salvos em 'stereoParams.npz'")
