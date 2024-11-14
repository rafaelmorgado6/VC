import cv2
import numpy as np
import math

# Carregar as imagens (src e dst) em grayscale
src = cv2.imread('/home/rafa/Desktop/VC/images/deti.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('/home/rafa/Desktop/VC/images/deti_tf.bmp', cv2.IMREAD_GRAYSCALE)

# Inicializar o detector ORB para extrair keypoints e descritores
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(src, None)
kp2, des2 = orb.detectAndCompute(dst, None)

# Criar objeto BFMatcher com distância Hamming e crossCheck habilitado
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Encontrar correspondências entre os descritores
matches = bf.match(des1, des2)

# Ordenar as correspondências pela distância (menor é melhor)
matches = sorted(matches, key=lambda x: x.distance)

# Selecionar as melhores correspondências (aqui usando 10%)
numGoodMatches = int(len(matches) * 0.5)
numGoodMatches = max(3, numGoodMatches)  # Garantir que tenhamos pelo menos 3 correspondências
matches = matches[:numGoodMatches]

# Desenhar as correspondências selecionadas
im_matches = cv2.drawMatches(src, kp1, dst, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Exibir as correspondências
cv2.imshow("Matches", im_matches)
cv2.waitKey(0)

# Preparar pontos de origem e destino a partir das correspondências
srcPts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dstPts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


# Convert selected points to numpy arrays
np_srcPts = np.array(srcPts).astype(np.float32)
np_dstPts = np.array(dstPts).astype(np.float32)

# Estimar a matriz de transformação afim que transforma srcPts em dstPts
affine_matrix = cv2.getAffineTransform(np_srcPts[:3], np_dstPts[:3])

# Imprimir a matriz de transformação afim estimada
print("Affine Transformation Matrix:")
print(affine_matrix)

# Warp the original image using the affine transformation
# Usar as dimensões da imagem de destino (dst) para a imagem deformada
warp_dst = cv2.warpAffine(src, affine_matrix, (dst.shape[1], dst.shape[0]))

# Display the warped image
cv2.imshow("Warped Image", warp_dst)
cv2.waitKey(0)

# Extract parameters from the affine matrix
a, c, tx = affine_matrix[0]
b, d, ty = affine_matrix[1]

# Compute translation
t_x = tx
t_y = ty

# Compute scaling factors
s_x = np.sqrt(a**2 + b**2)
s_y = np.sqrt(c**2 + d**2)

# Compute rotation angle
psi = math.atan2(b, a)

# Convert rotation angle from radians to degrees
rotation_angle = np.degrees(psi)

# Display computed parameters
print(f"Translation: t_x = {t_x}, t_y = {t_y}")
print(f"Scale: s_x = {s_x}, s_y = {s_y}")
print(f"Rotation angle (in degrees): {rotation_angle}")

# Compute the absolute difference between the two images
difference = cv2.absdiff(warp_dst, dst)

# Display the difference
cv2.imshow("Difference between Warped and Transformed", difference)
cv2.waitKey(0)
cv2.destroyAllWindows();
  


