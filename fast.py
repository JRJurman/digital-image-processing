"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""

import numpy as np
import cv2
import ipcv

def fast(src, differenceThreshold=50, contiguousThreshold=12, nonMaximalSuppression=True):
    """
    Function to determine corners in an image using the FAST corner detection
    algorithm.

    Args:
        src (array): image to find corners in
        differenceThreshold (optional[int]): how different (or similar) the
            brightness of a pixel should be, compared to it's center.
        contiguousThreshold (optional[int]): how many pixels should be
            contiguous in the radius of pixels.
        nonMaximalSuppression (optional[int]): whether or not to suppress
            groupings of points.

    Return:
        a ndarray of the shape of src, with values for how strong a corner was
        detected.
    """

    # first, build a 16 deep array of source images that represent the radius around
    # a center pixel
    srcImages = []
    # these are the offset from the center pixel
    # they are y-offset, x-offset
    radiusOffsets = np.array([
        [-3, 0],
        [-3, 1],
        [-2, 2],
        [-1, 3],
        [0, 3],
        [1, 3],
        [2, 2],
        [3, 1],
        [3, 0],
        [3, -1],
        [2, -2],
        [1, -3],
        [0, -3],
        [-1, -3],
        [-2, -2],
        [-3, -1]
    ])

    for i in range(radiusOffsets.shape[0]):
        offsetSrc = np.roll(src, radiusOffsets[i][0], axis=0)
        offsetSrc = np.roll(offsetSrc, radiusOffsets[i][1], axis=1)
        srcImages.append(offsetSrc)

    # turn our list of images into a 3D cube of pixels
    npSrcImages = np.array(srcImages)

    # result image that we'll put ranks into
    result = np.zeros(src.shape)
    for a in range(src.shape[0]):
        for b in range(src.shape[1]):
            center = src[a][b]
            # get the row of pixels
            radius =  npSrcImages[:,a,b]
            # double the row (so that we don't have to roll over it)
            # this is a cs trick for finding if a word is a shift of another word
            # ( for example, you can shift a word, like 'example', and append it to
            # itself: shift='ampleex' -> 'ampleexampleex' )
            doubleRadius = np.array([radius,radius]).flatten()

            # get all the brighter than pixels (using where)
            brightRadius = np.where(doubleRadius>(center+differenceThreshold), 1, 0)
            darkRadius = np.where(doubleRadius<(center-differenceThreshold), 1, 0)

            # split the array by 0s so we can find the 1 clusters
            brightSplit = np.split(brightRadius, np.where(brightRadius==0)[0])
            darkSplit = np.split(darkRadius, np.where(darkRadius==0)[0])

            # determine if contiguousThreshold exists
            brightCornerMax = np.array([len(e) for e in brightSplit]).max()
            darkCornerMax = np.array([len(e) for e in darkSplit]).max()

            brightCorner = brightCornerMax > contiguousThreshold
            darkCorner = darkCornerMax > contiguousThreshold

            if (brightCorner or darkCorner):
                result[a][b] = 1

    return result

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import os.path
    import time
    import cv2
    import ipcv
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + \
        'src/python/examples/data/sparse_checkerboard.tif'

    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    startTime = time.time()
    dst = ipcv.fast(src, differenceThreshold=50,
                  contiguousThreshold=9,
                  nonMaximalSuppression=True)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, src)

    if len(src.shape) == 2:
        annotatedImage = cv2.merge((src, src, src))
    else:
        annotatedImage = src
    annotatedImage[dst == 1] = [0,0,255]

    cv2.namedWindow(filename + ' (FAST Corners)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (FAST Corners)', annotatedImage)

    print('Corner coordinates ...')
    indices = numpy.where(dst == 1)
    numberCorners = len(indices[0])
    if numberCorners > 0:
        for corner in range(numberCorners):
            print('({0},{1})'.format(indices[0][corner], indices[1][corner]))

    action = ipcv.flush()
