import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/homography_4.jpg')

# Inicializar lista para armazenar os pontos selecionados pelo usuário
points = []

# Função de callback do mouse para capturar os pontos de canto
def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Corners", img)

# Exibir a imagem e definir o callback para capturar os pontos
cv2.imshow("Select Corners", img)
cv2.setMouseCallback("Select Corners", select_point)

# Aguardar até que 4 pontos sejam selecionados
while len(points) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()

# Converter a lista de pontos selecionados para um array numpy
src_points = np.float32(points)

# Definir os pontos de destino para a transformação de perspectiva (o tamanho do livro)
book_width = 235  # largura em pixels
book_height = 175  # altura em pixels

# Os pontos de destino correspondem ao tamanho do livro
dst_points = np.float32([
    [0, 0],
    [book_width, 0],
    [book_width, book_height],
    [0, book_height]])

# Calcular a matriz de homografia
M, mask = cv2.findHomography(src_points, dst_points)

# Aplicar a transformação de perspectiva com o tamanho da imagem original
warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Exibir a imagem corrigida
cv2.imshow("Corrected Perspective", warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

