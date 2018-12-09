import numpy as np
import scipy.misc
import cv2
#from scipy import ndimage
from datetime import datetime
import skimage.measure


# defining all required filters
sobel = {
        'horizontal': [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        'vertical': [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        }

prewitt = {
        'horizontal': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        'vertical': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        }

roberts = {
        'diagonal': [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
        'off-diagonal': [[0, 0, 0], [0, 0, 1], [0, -1, 0]]
        }


# convolution of image with filter/kernel
def convolve2d(image, kernel):

    output = np.zeros_like(image)
    image_padded = np.zeros((np.shape(image)[0] + np.shape(kernel)[0] - 1, np.shape(image)[1] + np.shape(kernel)[0] - 1))
    image_padded[1:-1, 1:-1] = image
    rows, columns = np.shape(image)
    for x in range(rows):
        for y in range(columns):
            output[x, y] = sum(sum(kernel*image_padded[x:x+3, y:y+3]))

    return output


# to save the filtered image
def write_image(image, name):

    outputDir = 'output/'
    output_image_name = outputDir + name + "_filtered_image_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_image_name, image)


# finding the magnitude of the image
def magnitude(horizontal_image, vertical_image):
    mag = np.hypot(horizontal_image, vertical_image)
    mag *= 255.0 / np.max(mag)

    return mag


# merged filter operations
def operators(image, hor_filter, ver_filter):
    # horizontal derivative using ready-made function
    # dx = ndimage.sobel(im, 0)
    dx = convolve2d(image, hor_filter)
    #write_image(dx, '_horizontal')

    # vertical derivative using ready-made function
    # dy = ndimage.sobel(im, 1)
    dy = convolve2d(image, ver_filter)
    #write_image(dy, '_vertical')

    # calculate magnitude of filtered images
    mag = magnitude(dx, dy)

    return mag


# adding filtered image to
def final_image(image, mag):

    return image + mag


def main(image, name, **weight):
    if len(image.shape) == 3:
        print("Color image uploaded. Converting to grayscale...")
        image = scipy.misc.imread(imagename, flatten=True)

    image = image.astype('int32')

    if name == 'sobel':
        mag = operators(image, hor_filter=sobel['horizontal'], ver_filter=sobel['vertical'])
    elif name == 'prewitt':
        mag = operators(image, hor_filter=prewitt['horizontal'], ver_filter=prewitt['vertical'])
    elif name == 'roberts':
        mag = operators(image, hor_filter=roberts['diagonal'], ver_filter=roberts['off-diagonal'])
    elif name == 'custom_sobel':
        custom_sobel = {
            'horizontal': [[1, weight['weight'], 1], [0, 0, 0], [-1, -weight['weight'], -1]],
            'vertical': [[1, 0, -1], [weight['weight'], 0, -weight['weight']], [1, 0, -1]]
        }

        mag = operators(image, hor_filter=custom_sobel['horizontal'], ver_filter=custom_sobel['vertical'])

    # save image in output folder
    #write_image(mag, name)

    final = final_image(image, mag)

    #write_image(final, name + '_final_image')

    return final


# finding psnr and mse values of filtered image
'''def mse_psnr_values(image, filtered_image):

    mse = skimage.measure.compare_mse(image, filtered_image)
    psnr = skimage.measure.compare_psnr(image, filtered_image)

    return [psnr, mse]'''


if __name__ == "__main__":
    imagename = 'images/cameraman.tiff'
    image = scipy.misc.imread(imagename)

    main(image, name='sobel', weight=30)

    '''# print the psnr and mse values
    psnr = mse_psnr_values(image, final)[0]
    mse = mse_psnr_values(image, final)[1]
    print("PSNR value: ", psnr)
    print("MSE value: ", mse)'''

