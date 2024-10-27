import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/art3.bmp', cv2.IMREAD_UNCHANGED )
image1 = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/art2.bmp', cv2.IMREAD_UNCHANGED )

if  np.shape(image) == () or np.shape(image1) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)
	

# Step 1: Convert to a binary image
_, binary_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
_, binary_image1 = cv2.threshold(image1, 120, 255, cv2.THRESH_BINARY)

# Step 2: Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)
inverted_image1 = cv2.bitwise_not(binary_image1)

# Step3: Criar um elemento estruturante circular
diameter = 11
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
kernel_sqr = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
kernel_sqr1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))


# Aplicar a dilatação na imagem negativa
eroded_image = cv2.erode(image, kernel)
opened_image = cv2.dilate(eroded_image, kernel)

eroded_image1 = cv2.erode(image1, kernel_sqr)
opened_image1 = cv2.dilate(eroded_image1, kernel_sqr)

eroded_image2 = cv2.erode(image1, kernel_sqr1)
opened_image2 = cv2.dilate(eroded_image1, kernel_sqr1)

# Criar subplots: 2 linhas e 3 colunas
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

# Primeira linha de imagens
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("art3.bmp")
axes[0, 1].imshow(image1, cmap="gray")
axes[0, 1].set_title("art2.bmp")
axes[0, 2].imshow(image1, cmap="gray")
axes[0, 2].set_title("art2.bmp")

# Segunda linha de imagens
axes[1, 0].imshow(eroded_image, cmap="gray")
axes[1, 0].set_title("Eroded Image art3.bmp")
axes[2, 0].imshow(opened_image, cmap="gray")
axes[2, 0].set_title("Opened image art3.bmp")
axes[1, 1].imshow(eroded_image1, cmap="gray")
axes[1, 1].set_title("Eroded Image (3,9) art2.bmp")
axes[2, 1].imshow(opened_image1, cmap="gray")
axes[2, 1].set_title("Opened  Image (3,9) art2.bmp")
axes[1, 2].imshow(eroded_image2, cmap="gray")
axes[1, 2].set_title("Eroded Image (9,3) art2.bmp")
axes[2, 2].imshow(opened_image2, cmap="gray")
axes[2, 2].set_title("Opened Image (9,3) art2.bmp")

# Ajustar os layouts para que os títulos não se sobreponham
plt.tight_layout()

# Exibir o plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()