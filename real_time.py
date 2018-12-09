import numpy as np
import scipy.misc
import cv2
#from scipy import ndimage
from datetime import datetime
import skimage.measure


################### image sharpening ###################
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
    write_image(dx, name + '_horizontal')

    # vertical derivative using ready-made function
    # dy = ndimage.sobel(im, 1)
    dy = convolve2d(image, ver_filter)
    write_image(dy, name + '_vertical')

    # calculate magnitude of filtered images
    mag = magnitude(dx, dy)

    return mag


# adding filtered image to
def final_image(image, mag):

    return image + mag


# finding psnr and mse values of filtered image
'''def mse_psnr_values(image, filtered_image):

    mse = skimage.measure.compare_mse(image, filtered_image)
    psnr = skimage.measure.compare_psnr(image, filtered_image)

    return [psnr, mse]'''


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, image = cap.read()

        # Our operations on the frame come here
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype('int32')

        name = 'sobel'

        if name == 'sobel':
            mag = operators(image, hor_filter=sobel['horizontal'], ver_filter=sobel['vertical'])
        elif name == 'prewitt':
            mag = operators(image, hor_filter=prewitt['horizontal'], ver_filter=prewitt['vertical'])
        elif name == 'custom_sobel':
            weight = {
                'weight': int(input('Enter weight: '))
            }

            custom_sobel = {
                'horizontal': [[1, weight['weight'], 1], [0, 0, 0], [-1, -weight['weight'], -1]],
                'vertical': [[1, 0, -1], [weight['weight'], 0, -weight['weight']], [1, 0, -1]]
            }

            mag = operators(image, hor_filter=custom_sobel['horizontal'], ver_filter=custom_sobel['vertical'])
        elif name == 'roberts':
            mag = operators(image, hor_filter=roberts['diagonal'], ver_filter=roberts['off-diagonal'])

        #final = final_image(image, mag)

        # Display the resulting frame
        cv2.imshow('frame', mag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

