# Aula_03_ex_01.py
#
# Historam visualization with openCV
#
# Paulo Dias

import cv2
import numpy as np
import sys

image_file = sys.argv[1]

# Função para desenhar a grade
def draw_grid(image, spacing=20):
    # Verifica se a imagem é em níveis de cinza ou colorida
    if len(image.shape) == 2:  # Imagem em níveis de cinza (2D array)
        grid_color = 255  # Branco
    else:  # Imagem colorida (3D array)
        grid_color = (128, 128, 128)  # Cinza

    # Obtém as dimensões da imagem
    height, width = image.shape[:2]

    # Desenha as linhas horizontais da grade
    for y in range(0, height, spacing):
        cv2.line(image, (0, y), (width, y), grid_color, 1)

    # Desenha as linhas verticais da grade
    for x in range(0, width, spacing):
        cv2.line(image, (x, 0), (x, height), grid_color, 1)

    return image

# Leitura da imagem do arquivo
image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

# Verifica se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem!")
    exit(-1)

# Aplica a grade na imagem
image_with_grid = draw_grid(image)

# Mostra a imagem resultante
cv2.imshow('Image with Grid', image_with_grid)

# Aguarda uma tecla para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
