import cv2
import numpy as np

# Carrega a imagem
image = cv2.imread( '/home/rafa/Desktop/ua_computerVision/images/Lena_Ruido.png', cv2.IMREAD_GRAYSCALE );

if image is None:
    print("Erro ao carregar a imagem.")

ksize1  = (3,3)
ksize2  = (5,5)
ksize3  = (7,7)
sigmax = 1.5

resize_dim = (700, 500)  # Defina o tamanho desejado
image_resized = cv2.resize(image, resize_dim)

# Aplica o filtro mediano com diferentes tamanhos de kernel
filtered_img1 = cv2.GaussianBlur(image_resized, ksize1, sigmax)
filtered_img2 = cv2.GaussianBlur(image_resized, ksize2, sigmax)
filtered_img3 = cv2.GaussianBlur(image_resized, ksize3, sigmax)

# Exibe a imagem original
cv2.imshow('Original', image_resized)
cv2.imshow('filtered_img1', filtered_img1)
cv2.imshow('filtered_img2', filtered_img2)
cv2.imshow('filtered_img3', filtered_img3)


cv2.waitKey(0)
cv2.destroyAllWindows()