import numpy as np
import cv2

def main():
    # Load the images
    #image = cv2.imread(r'C:\Users\Rafa\PycharmProjects\VC\deti.bmp',  cv2.IMREAD_UNCHANGED)
    image = cv2.imread(r'C:\Users\Rafa\PycharmProjects\VC\input.png',  cv2.IMREAD_UNCHANGED)

    # Check if images were loaded correctly
    if image is None:
        print("Error: One of the images could not be loaded.")
        return

    # Display the image
    cv2.imshow("input.png", image)

    # Size
    histSize = 256  # from 0 to 255
    # Intensity Range
    histRange = [0, 256]

    # Compute the histogram
    hist_item = cv2.calcHist([image], [0], None, [histSize], histRange)

    ##########################################
    # Drawing with openCV
    # Create an image to display the histogram
    histImageWidth = 512
    histImageHeight = 512
    color = 125
    histImage = np.zeros((histImageWidth, histImageHeight, 1), np.uint8)

    # Width of each histogram bar
    binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

    # Normalize values to [0, histImageHeight]
    cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)

    # Draw the bars of the nomrmalized histogram
    for i in range(histSize):
        cv2.rectangle(histImage, (i * binWidth, 0), ((i + 1) * binWidth, int(hist_item[i])), 125, -1)

    # ATTENTION : Y coordinate upside down
    histImage = np.flipud(histImage)

    cv2.imshow('colorhist input.png', histImage)
    cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
