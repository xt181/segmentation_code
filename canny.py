import cv2

# Load the image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 100, 20)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

