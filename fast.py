"""
PYTHON METHOD DEFINITION
"""
def fast(src, differenceThreshold=50, contiguousThreshold=12, nonMaximalSuppression=True):
    pass


"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import os.path
    import time

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
