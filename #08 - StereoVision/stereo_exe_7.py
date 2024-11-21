import numpy as np
import cv2
import glob

# Carregar os parâmetros de calibração salvos
with np.load("stereoParams.npz") as data:
    intrinsics1 = data['intrinsics1']
    distortion1 = data['distortion1']
    intrinsics2 = data['intrinsics2']
    distortion2 = data['distortion2']
    R = data['R']  # Matriz de rotação entre câmeras
    T = data['T']  # Vetor de translação entre câmeras

# Selecionar um par de imagens estéreo (esquerda e direita)
left_images = sorted(glob.glob('..//images//left*.jpg'))
right_images = sorted(glob.glob('..//images//right*.jpg'))

# Usar o primeiro par de imagens para demonstração
if not left_images or not right_images:
    raise FileNotFoundError("Imagens de calibração não encontradas.")

left_img = cv2.imread(left_images[0])
right_img = cv2.imread(right_images[0])

# Dimensões da imagem
height, width = left_img.shape[:2]

# Matrizes de retificação e projeção
R1 = np.zeros((3, 3))
R2 = np.zeros((3, 3))
P1 = np.zeros((3, 4))
P2 = np.zeros((3, 4))
Q = np.zeros((4, 4))

# Calcular as matrizes de retificação
cv2.stereoRectify(
    intrinsics1, distortion1, intrinsics2, distortion2,
    (width, height), R, T, R1, R2, P1, P2, Q,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0)
)

# Computar os mapas de retificação para ambas as câmeras
map1x, map1y = cv2.initUndistortRectifyMap(intrinsics1, distortion1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(intrinsics2, distortion2, R2, P2, (width, height), cv2.CV_32FC1)

# Aplicar a transformação de retificação usando cv2.remap
rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

# Conversão para escala de cinza
gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

# Inicializar o algoritmo StereoBM
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)

# Calcular o mapa de disparidade
disparity = stereo.compute(gray_left, gray_right)

# Normalizar o mapa de disparidade para exibição
disparity_normalized = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Calcular as coordenadas 3D dos pixels
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Salvar as coordenadas 3D em um arquivo .npz
np.savez("points_3D.npz", points_3D=points_3D)

# Exibir o mapa de disparidade
cv2.imshow("Mapa de Disparidade", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
