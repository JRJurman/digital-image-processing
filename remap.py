"""
IMPORTS
"""
import numpy as np
import ipcv
import cv2

"""
PYTHON METHOD DEFINITION
"""
def remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=0):
    """
    Function to transform the src image using the maps

    Args:
        src (array): source image
        map1 (array): the first map, the x value for a given coord
        map2 (array): the second map, the y value for a given coord
        interpolation (optional[int]): type of interpolation to use
            defaults to Nearest-neighbor
        borderMode (optional[int]): type of border to include outside the image
            defaults to constant value
        borderValue (optional[int]): border value for constant value border
            defaults to 0 (black)

    Returns:
        the source modified to the new mappings
    """
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
    filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    src = cv2.imread(filename)

    map1, map2 = ipcv.map_rotation_scale(src, rotation=30, scale=[1.3, 0.8])

    startTime = time.clock()
    dst = ipcv.remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=0)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))

    srcName = 'Source (' + filename + ')'
    cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(srcName, src)

    dstName = 'Destination (' + filename + ')'
    cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(dstName, dst)

    ipcv.flush()
