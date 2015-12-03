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

    # double the row (so that we don't have to roll over it)
    # this is a cs trick for finding if a word is a shift of another word
    # ( for example, you can shift a word, like 'example', and append it to
    # itself: shift='ampleex' -> 'ampleexampleex' )
    for image in srcImages[:]:
        srcImages.append(image)

    # turn our list of images into a 3D cube of pixels
    npSrcImages = np.array(srcImages)

    # determine where a bright radius or dark radius exists
    brightRadiusImages = np.where(npSrcImages>(src+differenceThreshold), 1, 0)
    darkRadiusImages = np.where(npSrcImages<(src-differenceThreshold), 1, 0)

    # swap the axes of the images so that we are looking at rows of neighbors
    brightRadiusRows = brightRadiusImages.swapaxes(0, 2)
    darkRadiusRows = darkRadiusImages.swapaxes(0, 2)

    # build every permutation of a possible corner
    # i.e. 1 1 0 0
    #      0 1 1 0
    #      0 0 1 1
    # we have to build this array with a for loop
    conChecks = []
    for i in range(len(radiusOffsets)):
        conChecks.append(np.array(
            [1]*contiguousThreshold + [0]*(contiguousThreshold-(2*len(radiusOffsets)))
        ).roll(i))

    # turn it into an np array of arrays
    npConChecks = np.array(conChecks)

    """
    # do logical ands of the rows against the table
    # get the sum of true values for those logical ands
    # and get the max value out of those values
    #  row    | table (conCheck) | Logical And | sum | max
    #         |  1 1 0 0         | F T F F     | 1   |
    # 0 1 1 0 |  0 1 1 0         | F T T F     | 2   |  2
    #         |  0 0 1 1         | F F T F     | 1   |

    np.sum(np.logical_and(brightRadiusRows, npConChecks), axis=1).max()
    """

    # we check sections of the rows
    for i in range(len(npConChecks)):
        np.max(
            np.sum(
                np.logical_and(
                    brightRadiusRows, npConChecks[i]),
                axis=2) # for sum
            axis=1) # for max



    """
    if (nonMaximalSuppression):
        for a in range(1, src.shape[0]-1):
            for b in range(1, src.shape[1]-1):

                # set the center pixel at a, b
                center = result[a][b]

                # get the neighbor of pixels
                neighbors = np.array([
                    result[a-1][b-1],
                    result[a-1][b],
                    result[a-1][b+1],
                    result[a][b-1],
                    result[a][b+1],
                    result[a+1][b-1],
                    result[a+1][b],
                    result[a+1][b+1]
                ])

                # determine the largest canidant for a corner
                maxNeighbor = neighbors.max()
                if center < maxNeighbor:
                    result[a][b] = 0

    return result >= 1
    """

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
