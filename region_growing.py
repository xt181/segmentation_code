import cv2
import numpy as np

def region_growing(image, seed, threshold= 78):
    # Create a mask and initialize it with zeros, with the same size as the input image
    rows, cols = image.shape
    segmented = np.zeros_like(image)

    # Create a queue for storing the pixels to be processed
    queue = []
    queue.append(seed)

    # Define the neighborhood connectivity (8-connectivity)
    connectivity = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]

    # Region growing loop
    while len(queue) > 0:
        # Get the current pixel from the queue
        current_pixel = queue.pop(0)
        x, y = current_pixel

        # Check if the current pixel is within the image bounds
        if x < 0 or y < 0 or x >= rows or y >= cols:
            continue

        # Check if the current pixel has already been visited
        if segmented[x, y] != 0:
            continue

        # Check if the intensity difference between the current pixel and the seed pixel is below the threshold
        if abs(int(image[x, y]) - int(image[seed])) > threshold:
            continue

        # Add the current pixel to the segmented region
        segmented[x, y] = 255

        # Add the neighbors of the current pixel to the queue
        for dx, dy in connectivity:
            queue.append((x + dx, y + dy))

    return segmented

# Load the image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

height, width = image.shape[:2]


# Define seed point
seed = (100, 100)

# Apply region growing
segmented = region_growing(image, seed)

roi = segmented[0:int(height/5.76), :]

roi[:, :] = 0

# Display the segmented image
cv2.imshow('Segmented Image', segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

