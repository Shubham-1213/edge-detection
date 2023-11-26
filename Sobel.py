import numpy as np
import cv2
import matplotlib.pyplot as plt


def sobel_edge_detection(image, verbose=False):
    # 5x5 Sobel kernels for edge detection
    sobel_x = np.array([[-2, -1, 0, 1, 2],
                        [-2, -1, 0, 1, 2],
                        [-4, -2, 0, 2, 4],
                        [-2, -1, 0, 1, 2],
                        [-2, -1, 0, 1, 2]])

    sobel_y = sobel_x.T  # Transpose to get the vertical edge detection kernel

    new_image_x = convolution(image, sobel_x, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, sobel_y, verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude


def convolution(image, kernel, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = kernel_row // 2
    pad_width = kernel_col // 2

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


if __name__ == '__main__':
    abs_path = 'C:\\Coding\\IVP\\ivp_project_edge_detection\\download.jpeg'

    image = cv2.imread(abs_path)

    sobel_edge_detection(image, verbose=True)
