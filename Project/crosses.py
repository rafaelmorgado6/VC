import cv2
import numpy as np

def enhance_image(image):

    # Amplify the image by resizing it (scaling up by 2x)
    height, width = image.shape[:2]
    enlarged = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Strengthen edges using dilation
    kernel = np.ones((3, 3), np.uint8)
    enhanced = cv2.dilate(enhanced, kernel, iterations=1)

    return enhanced

def detect_and_circle_crosses(enhanced_image):
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(enhanced_image, 30, 100)  # Adjust thresholds

    # Detect contours to simplify the detection of regions with lines
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected = False  # Initialize detection status

    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Get the bounding rectangle for the contour
        (x, y, w, h) = cv2.boundingRect(approx)

        # Use heuristics to identify possible crosses
        if w > 10 and h > 10 and 0.8 < float(w) / h < 1.2:  # Ensure approximate square aspect ratio
            # Draw a circle around the detected region
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(image, (center_x, center_y), max(w, h) // 2, (0, 255, 0), 3)
            detected = True  # Update detection status

    return image, detected

# Main Program
if __name__ == "__main__":
    # Load the image
    image_path = '/home/rafa/Desktop/VC/Project/teste.png'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image. Please check the file path.")
    else:
        # Process the image and check detection status
        enhanced_image = enhance_image(image)
        processed_image, is_cross_detected = detect_and_circle_crosses(enhanced_image)

        print("Cross detected:", is_cross_detected)

        # Display the processed image
        cv2.imshow("Original Image", image)
        cv2.imshow("Enhanced Crosses", enhanced_image)
        cv2.imshow("Detected Crosses", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
