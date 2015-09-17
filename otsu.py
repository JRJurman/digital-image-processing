"""
PYTHON METHOD DEFINITION
"""
def otsu_threshold(im, maxCount=255, verbose=False):
    pass

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'

    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(filename))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    startTime = time.time()
    thresholdedImage, threshold = ipcv.otsu_threshold(im, verbose=True)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    print('Threshold = {0}'.format(threshold))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, im)
    cv2.namedWindow(filename + ' (Thresholded)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Thresholded)', thresholdedImage * 255)

    action = ipcv.flush()
