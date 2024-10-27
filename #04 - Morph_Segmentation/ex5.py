import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/art4.bmp', cv2.IMREAD_UNCHANGED )

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)
	
#cv2.imshow("Original Image", image)

# Step 1: Convert to a binary image
_, binary_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

# Step 2: Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)

# Step3: Criar um elemento estruturante circular
diameter = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


# Aplicar a dilatação na imagem negativa
dilated_image = cv2.dilate(image, kernel)
closed_image = cv2.erode(dilated_image, kernel)


# Criar subplots: 2 linhas e 3 colunas
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

# Primeira linha de imagens
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(inverted_image, cmap="gray")
axes[1].set_title("Inverted Image")
axes[2].imshow(closed_image, cmap="gray")
axes[2].set_title("Closed Image")


# Ajustar os layouts para que os títulos não se sobreponham
plt.tight_layout()

# Exibir o plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()