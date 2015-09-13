"""
PYTHON METHOD DEFINITION
"""
def histogram_enhancement(im, etype='linear2', target=None, maxCount=255):
    return im
"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time
    # should include numpy import? ============================================#
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    # should be this? =========================================================#
    img = "giza.jpg"
    filename = os.path.join(home, 'src', 'python', 'examples', 'data', img)

    matchFilename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
    matchFilename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    matchFilename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
    matchFilename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    # should be this? =========================================================#
    img = "crowd.jpg"
    matchFilename = os.path.join(home, 'src', 'python', 'examples', 'data', img)

    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(filename))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)

    print('Linear 2% ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='linear2')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 2%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 2%)', enhancedImage)

    print('Linear 1% ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='linear1')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Linear 1%)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Linear 1%)', enhancedImage)

    print('Equalized ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='equalize')
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Equalized)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Equalized)', enhancedImage)

    tgtIm = cv2.imread(matchFilename, cv2.IMREAD_UNCHANGED)
    print('Matched (Image) ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtIm)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Image)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Image)', enhancedImage)

    tgtPDF = numpy.ones(256) / 256
    print('Matched (Distribution) ...')
    startTime = time.time()
    enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtPDF)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))
    cv2.namedWindow(filename + ' (Matched - Distribution)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Matched - Distribution)', enhancedImage)

    action = ipcv.flush()
