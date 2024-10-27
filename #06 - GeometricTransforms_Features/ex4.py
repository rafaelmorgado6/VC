import cv2
import numpy as np

# Carregar as imagens (src e dst) em grayscale
src = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/deti.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/deti_tf.bmp', cv2.IMREAD_GRAYSCALE)

# Inicializar o detector ORB para extrair keypoints e descritores
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(src, None)
kp2, des2 = orb.detectAndCompute(dst, None)

# Criar objeto BFMatcher com distância L2 e crossCheck habilitado
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Encontrar correspondências entre os descritores
matches = bf.match(des1, des2)

# Ordenar as correspondências pela distância (menor é melhor)
matches = sorted(matches, key=lambda x: x.distance)

# Selecionar as melhores correspondências (aqui usando 10%)
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Desenhar as correspondências selecionadas
im_matches = cv2.drawMatches(src, kp1, dst, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Exibir as correspondências
cv2.imshow("Matches", im_matches)

# Preparar pontos de origem e destino a partir das correspondências
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Esperar tecla para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
