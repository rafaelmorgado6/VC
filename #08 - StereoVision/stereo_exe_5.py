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

# Função de callback para manipular eventos do mouse
def mouse_handler_left(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Exibe linha epipolar correspondente na imagem direita
        print(f"Clique detectado na esquerda em y = {y}")
        # Desenhar a linha epipolar na imagem direita
        color = (0, 255, 0)  # Cor verde para a linha
        cv2.line(rectified_right, (0, y), (width, y), color, 1)
        cv2.imshow("Imagem Direita Retificada", rectified_right)

def mouse_handler_right(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Exibe linha epipolar correspondente na imagem esquerda
        print(f"Clique detectado na direita em y = {y}")
        # Desenhar a linha epipolar na imagem esquerda
        color = (0, 255, 0)  # Cor verde para a linha
        cv2.line(rectified_left, (0, y), (width, y), color, 1)
        cv2.imshow("Imagem Esquerda Retificada", rectified_left)

# Desenhar linhas horizontais para verificação de alinhamento
spacing = 25
for y in range(0, height, spacing):
    color = (255, 0, 0)  # Cor azul para linhas iniciais
    cv2.line(rectified_left, (0, y), (width, y), color, 1)
    cv2.line(rectified_right, (0, y), (width, y), color, 1)

# Exibir as imagens retificadas e associar o callback de mouse
cv2.imshow("Imagem Esquerda Retificada", rectified_left)
cv2.setMouseCallback("Imagem Esquerda Retificada", mouse_handler_left)

cv2.imshow("Imagem Direita Retificada", rectified_right)
cv2.setMouseCallback("Imagem Direita Retificada", mouse_handler_right)

# Aguardar a interação do usuário para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()
