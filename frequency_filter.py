"""
PYTHON METHOD DEFINITION
Jesse Jurman (jrj2703)
"""

import numpy as np
import cv2

def frequency_filter(im, frequencyFilter, delta=0):
    """
    Function to apply a filter to an image.

    Args:
        im (array): source image to be filtered
            if the image is color, it will be converted to grayscale
        frequencyFilter (int): the frequency to filter the images on
        delta (optional[int]): value to increase the digital counts by

    Return:
        the image with the applied filter, as numpy.complex128 data types
    """
    if(len(im.shape) == 3):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # shift image to line up with fourier transformation
    filterShift = np.fft.fftshift(frequencyFilter)

    # compute the fourier transform using fft
    fourierTransform = np.fft.fft2(im)

    # multiple the resulting fourier transform by the shifted filter
    appliedFilter = filterShift * fourierTransform

    # compute the inverse transform of the fourier transform
    filteredImage = np.fft.ifft2(appliedFilter)

    # display the magnitude to see the enhanced image
    magnitude = 20*np.log(np.abs(filteredImage))
    return magnitude

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import numpy
    import os.path
    import time

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'

    im = cv2.imread(filename)

    frequencyFilter = ipcv.filter_lowpass(im,
                                       16,
                                       filterShape=ipcv.IPCV_GAUSSIAN)

    startTime = time.clock()
    offset = 0
    filteredImage = ipcv.frequency_filter(im, frequencyFilter, delta=offset)
    filteredImage = numpy.abs(filteredImage)
    filteredImage = filteredImage.astype(dtype=numpy.uint8)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (frequency_filter)= {0} [s]'.format(elapsedTime))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)
    cv2.imshow(filename, ipcv.histogram_enhancement(im))

    filterName = 'Filtered (' + filename + ')'
    cv2.namedWindow(filterName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filterName, filteredImage)
    cv2.imshow(filterName, ipcv.histogram_enhancement(filteredImage))

    ipcv.flush()
