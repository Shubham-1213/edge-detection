import cv2
import numpy as np

# Read the image
abs_path = 'C:\\Coding\\IVP\\ivp_project_edge_detection\\download.jpeg'

image = cv2.imread(abs_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur kernel size and sigma
kernel_size = 5
sigma = 1.4

# Define a Gaussian kernel function
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Convolution function to apply a kernel on an image
def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, ((pad_amount, pad_amount), (pad_amount, pad_amount)), mode='constant')

    result = np.zeros_like(image)
    
    for y in range(image_height):
        for x in range(image_width):
            result[y, x] = np.sum(kernel * padded_image[y:y+kernel_size, x:x+kernel_size])

    return result

# Create Gaussian kernel
gaussian_kernel_array = gaussian_kernel(kernel_size, sigma)

# Apply convolution to blur the image
blurred_image = convolution(gray_image, gaussian_kernel_array)

# Define the Laplacian filter
laplacian_filter = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply convolution with the Laplacian filter
filtered_image = convolution(blurred_image, laplacian_filter)

# Convert the output to 8-bit image
filtered_image = np.uint8(np.absolute(filtered_image))

# Show the original, blurred, and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Laplacian Edge Detection', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
