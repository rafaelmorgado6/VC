import numpy as np
import cv2

def main():
    # Load the images
    #image = cv2.imread(r'C:\Users\Rafa\PycharmProjects\VC\deti.bmp',  cv2.IMREAD_UNCHANGED)
    image = cv2.imread(r'/home/rafa/Desktop/ua_computerVision/images/input.png',  cv2.IMREAD_UNCHANGED)

    # Check if images were loaded correctly
    if image is None:
        print("Error: One of the images could not be loaded.")
        return

    cv2.imshow("input.png", image)

    # Find min and max values of the histogram
    min_val, max_val, _, _ = cv2.minMaxLoc(image)

    # Create a new image using the formula g = (f - fmin) / (fmax - fmin)
    new_image = (((image - min_val) / (max_val - min_val))* 255).astype(np.uint8)

    # Display the contrast-stretched image
    cv2.imshow("Contrast-Stretched Image", new_image)

    # Size
    histSize = 256  # from 0 to 255
    histRange = [0, 256]
    
    # Compute histograms
    hist_item1 = cv2.calcHist([image], [0], None, [histSize], histRange)
    hist_item2 = cv2.calcHist([new_image], [0], None, [histSize], histRange)

    ##########################################
    # Drawing with openCV
    # Create an image to display the histogram
    histImageWidth = 512
    histImageHeight = 512
    color = 125
    
    histImage = np.zeros((histImageWidth, histImageHeight, 1), np.uint8)
    histnew_Image = np.zeros((histImageWidth, histImageHeight, 1), np.uint8)

    # Width of each histogram bar
    binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

    # Normalize values to [0, histImageHeight]
    cv2.normalize(hist_item1, hist_item1, 0, histImageHeight, cv2.NORM_MINMAX)
    cv2.normalize(hist_item2, hist_item2, 0, histImageHeight, cv2.NORM_MINMAX)

    # Draw the bars of the nomrmalized histogram
    for i in range(histSize):
        cv2.rectangle(histImage, (i * binWidth, 0), ((i + 1) * binWidth, int(hist_item1[i])), 125, -1)
        cv2.rectangle(histnew_Image, (i * binWidth, 0), ((i + 1) * binWidth, int(hist_item2[i])), 125, -1)
    
    # ATTENTION : Y coordinate upside down
    histImage = np.flipud(histImage)
    histnew_Image = np.flipud(histnew_Image)

    # Display the image

    cv2.imshow('colorhist input.png', histImage)
    cv2.imshow('New colorhist input.png', histnew_Image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
