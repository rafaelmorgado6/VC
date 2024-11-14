import cv2
import numpy as np

# Dimensões do livro
book_width_cm = 17.5
book_height_cm = 23.5


def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
        cv2.imshow("Select Corners", image)


image = cv2.imread('/home/rafa/Desktop/VC/images/homography_4.jpg')
width, height, _ = image.shape
cv2.imshow("Select Corners", image)

# Lista de pontos selecionados
points = []
cv2.setMouseCallback("Select Corners", select_points)

# Aguarda a seleção de quatro pontos
print("Selecione os quatro cantos do livro na imagem.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Verifica se quatro pontos foram selecionados
if len(points) != 4:
    print("Erro: Selecione exatamente quatro pontos.")
else:
    # Define os pontos de destino com base no tamanho do livro
    destination_points = np.array([
        [0, 0],
        [book_width_cm * 10, 0],
        [book_width_cm * 10, book_height_cm * 10],
        [0, book_height_cm * 10]
    ], dtype="float32")

    # Converte os pontos selecionados em uma matriz numpy
    src_points = np.array(points, dtype="float32")

    # Calcula a homografia
    h_matrix, _ = cv2.findHomography(src_points, destination_points)

    warped_image = cv2.warpPerspective(image, h_matrix, (width, height))

    # Exibe a imagem corrigida
    cv2.imshow("Corrected Homography", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()