import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/lena.jpg', cv2.IMREAD_UNCHANGED );

if np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)
	
# Image characteristics
if len (image.shape) > 2:
	print ("The loaded image is NOT a GRAY-LEVEL image !")
	exit(-1)

threshold_value =127
# Aplica os diferentes tipos de thresholding
_, thresh_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO)
_, thresh_tozero_inv = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO_INV)

# Criar subplots: 2 linhas e 3 colunas
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Primeira linha de imagens
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("image")
axes[0, 1].imshow(thresh_binary, cmap="gray")
axes[0, 1].set_title("thresh_binary")
axes[0, 2].imshow(thresh_binary_inv, cmap="gray")
axes[0, 2].set_title("thresh_binary_inv")

# Segunda linha de imagens
axes[1, 0].imshow(thresh_trunc, cmap="gray")
axes[1, 0].set_title("thresh_trunc")
axes[1, 1].imshow(thresh_tozero, cmap="gray")
axes[1, 1].set_title("thresh_tozero")
axes[1, 2].imshow(thresh_tozero_inv, cmap="gray")
axes[1, 2].set_title("thresh_tozero_inv")


# Ajustar os layouts para que os títulos não se sobreponham
plt.tight_layout()

# Exibir o plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()