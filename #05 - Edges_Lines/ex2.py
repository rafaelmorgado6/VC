import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)

# Read the image from argv
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
image = cv2.imread( '/home/rafa/Desktop/ua_computerVision/images/Lena_Ruido.png', cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

# Average filter 3 x 3
imageAFilter3x3_1 = cv2.blur( image, (3, 3))
imageAFilter3x3_2 = cv2.blur( imageAFilter3x3_1, (3, 3))
imageAFilter3x3_3 = cv2.blur( imageAFilter3x3_2, (3, 3))

imageAFilter5x5_1 = cv2.blur( image, (5, 5))
imageAFilter5x5_2 = cv2.blur( imageAFilter5x5_1, (5, 5))
imageAFilter5x5_3 = cv2.blur( imageAFilter5x5_2, (5, 5))

imageAFilter7x7_1 = cv2.blur( image, (7, 7))
imageAFilter7x7_2 = cv2.blur( imageAFilter7x7_1, (7, 7))
imageAFilter7x7_3 = cv2.blur( imageAFilter7x7_2, (7, 7))

# Criar subplots: 2 linhas e 3 colunas
fig, axes = plt.subplots(4, 3, figsize=(12, 8))

# Primeira linha de imagens
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("image")
axes[0, 1].set_visible(False)
axes[0, 2].set_visible(False)

# Segunda linha de imagens
axes[1, 0].imshow(imageAFilter3x3_1, cmap="gray")
axes[1, 0].set_title("Average Filter 3 x 3 - 1 Iter")
axes[1, 1].imshow(imageAFilter3x3_2, cmap="gray")
axes[1, 1].set_title("Average Filter 3 x 3 - 2 Iter")
axes[1, 2].imshow(imageAFilter3x3_3, cmap="gray")
axes[1, 2].set_title("Average Filter 3 x 3 - 3 Iter")

# Terceira linha de imagens
axes[2, 0].imshow(imageAFilter5x5_1, cmap="gray")
axes[2, 0].set_title("Average Filter 5 x 5 - 1 Iter")
axes[2, 1].imshow(imageAFilter5x5_2, cmap="gray")
axes[2, 1].set_title("Average Filter 5 x 5 - 2 Iter")
axes[2, 2].imshow(imageAFilter5x5_3, cmap="gray")
axes[2, 2].set_title("Average Filter 5 x 5 - 3 Iter")

# Quarta linha de imagens
axes[3, 0].imshow(imageAFilter7x7_1, cmap="gray")
axes[3, 0].set_title("Average Filter 7 x 7 - 1 Iter")
axes[3, 1].imshow(imageAFilter7x7_2, cmap="gray")
axes[3, 1].set_title("Average Filter 7 x 7 - 2 Iter")
axes[3, 2].imshow(imageAFilter7x7_3, cmap="gray")
axes[3, 2].set_title("Average Filter 7 x 7 - 3 Iter")


# Ajustar os layouts para que os títulos não se sobreponham
plt.tight_layout()

# Exibir o plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
