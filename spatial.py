"""
PYTHON METHOD DEFINITION
"""

import numpy as np

def filter2D(src, dstDepth, kernel, delta=0, maxCount=255):
    """
    function to filter an image using a kernel

    Args:
        src (array): image to be filtered
        dstDepth (int): type of image to return
        kernel (array): filter to convolve with
        delta (optional[int]): gray value offset to increase digital counts
            default 0
        maxCount (optional[int]): the maximum value for a digital count
            default 255

    Return:
        the quantized image
    """
    # define a center to the kernel
    center = np.floor(np.array(kernel.shape)/2).astype(dstDepth)

    # average kernel into percentages
    kernel = kernel.astype(np.float64)
    kernel /= kernel.sum()
    print(kernel)

    # generate rolled images
    imageArray = []
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            copy = np.copy(src).astype(np.float64)
            # manipulate each copy image
            copy *= kernel[row][col]
            # roll the image
            copy = np.roll(copy, row-center[0], 0)
            copy = np.roll(copy, col-center[1], 1)
            # push onto our array
            imageArray.append(copy)

    # add them together
    resultImage = np.zeros(src.shape)
    while(len(imageArray) > 1):
        resultImage += imageArray.pop()

    return np.array(resultImage+delta, dstDepth)

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import os.path
    import time
    import ipcv
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'

    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    dstDepth = ipcv.IPCV_8U
    # kernel = numpy.asarray([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # offset = 0
    kernel = numpy.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    offset = 128
    # kernel = numpy.ones((15,15))
    # offset = 0
    # kernel = numpy.asarray([[1,1,1],[1,1,1],[1,1,1]])
    # offset = 0


    startTime = time.time()
    dst = ipcv.filter2D(src, dstDepth, kernel, delta=offset)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, src)

    cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Filtered)', dst)

    action = ipcv.flush()
