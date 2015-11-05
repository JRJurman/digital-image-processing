"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""
import numpy as np
import cv2

def Ix(src):
    """
    Function to return the first partial derivitive of a source image, with
    respect to the horizontal.
    """
    kernel = np.array([-1, 0, 1])
    return cv2.filter2D(src, -1, kernel)

def Iy(src):
    """
    Function to return the first partial derivitive of a source image, with
    respect to the vertical.
    """
    kernel = np.array([-1, 0, 1]).reshape(1, 3)
    return cv2.filter2D(src, -1, kernel)

def harris(src, sigma=1, k=0.04):
    """
    Function to determine corners in an image using the harris corner detection
    algorithm.

    Args:
        src (array): image to find corners in
        sigma (optional[int]): size of gaussian (delta x and delta y)
            defaults to 1
        k (optional[float]): the harris constant
            defaults to 0.04

    Return:
        a ndarray of the shape of src, with values for how strong a corner was
        detected.
    """

    srcDtype = src.dtype
    src = src.astype(np.float64)

    partialDerX = Ix(src)
    partialDerY = Iy(src)

    A = partialDerX**2
    B = partialDerY**2
    C = partialDerX * partialDerY

    # convolution with gaussian to reduce the effect of noise
    kernelX = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    kernelY = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    kernel = np.exp(-0.5 * (((kernelX**2)+(kernelY**2))/(sigma**2)))

    A = cv2.filter2D(A, -1, kernel)
    B = cv2.filter2D(B, -1, kernel)
    C = cv2.filter2D(C, -1, kernel)

    Tr = A + B
    Det = (A*B) - C**2
    R = Det - k*(Tr**2)

    return R

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import os.path
    import time
    import ipcv
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + \
            'src/python/examples/data/sparse_checkerboard.tif'

    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    sigma = 1
    k = 0.04
    startTime = time.time()
    dst = ipcv.harris(src, sigma, k)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, src)

    if len(src.shape) == 2:
        annotatedImage = cv2.merge((src, src, src))
    else:
        annotatedImage = src
    fractionMaxResponse = 0.25
    annotatedImage[dst > fractionMaxResponse*dst.max()] = [0,0,255]

    cv2.namedWindow(filename + ' (Harris Corners)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Harris Corners)', annotatedImage)

    print('Corner coordinates ...')
    indices = numpy.where(dst > fractionMaxResponse*dst.max())
    numberCorners = len(indices[0])
    if numberCorners > 0:
        for corner in range(numberCorners):
           print('({0},{1})'.format(indices[0][corner], indices[1][corner]))

    action = ipcv.flush()
