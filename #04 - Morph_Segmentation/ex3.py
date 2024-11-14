import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/mon1.bmp', cv2.IMREAD_UNCHANGED )

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)
	
#cv2.imshow("Original Image", image)

# Step 1: Convert to a binary image
_, binary_image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)

# Step 2: Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)

# Step3: Criar um elemento estruturante circular
diameter = 11
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
kernel_sqr = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


# Aplicar a dilatação na imagem negativa
eroded_image = cv2.erode(inverted_image, kernel)
eroded_image1 = cv2.erode(eroded_image, kernel_sqr)
eroded_sqr_image1 = cv2.erode(inverted_image, kernel_sqr)
eroded_sqr_image2 = cv2.erode(eroded_sqr_image1, kernel_sqr)

# Criar subplots: 2 linhas e 3 colunas
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

# Primeira linha de imagens
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(binary_image, cmap="gray")
axes[0, 1].set_title("Binary Image")
axes[0, 2].imshow(inverted_image, cmap="gray")
axes[0, 2].set_title("Inverted Image")

# Segunda linha de imagens
axes[1, 0].imshow(eroded_image, cmap="gray")
axes[1, 0].set_title("Eroded Image 1")
axes[1, 1].imshow(eroded_image1, cmap="gray")
axes[1, 1].set_title("Eroded Image 2")
axes[2, 0].imshow(eroded_sqr_image1, cmap="gray")
axes[2, 0].set_title("Eroded Square Image 1")
axes[2, 1].imshow(eroded_sqr_image2, cmap="gray")
axes[2, 1].set_title("Eroded Square Image 2")

# Remover subplots extras que não têm imagens
axes[1, 2].axis('off')
axes[2, 2].axis('off')

# Ajustar os layouts para que os títulos não se sobreponham
plt.tight_layout()

# Exibir o plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()