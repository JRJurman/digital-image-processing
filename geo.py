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

"""
PYTHON METHOD DEFINITION
"""
def map_rotation_scale(src, rotation=0, scale=[1, 1]):
    """
    Function to rotate or scale image

    Args:
        src (array): source image to transform
        rotation (int): degrees to rotate image by
        scale (array): factors to scale image by in the x (0th) and y (1st)

    Returns:
        a transformed image
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

    startTime = time.clock()
    map1, map2 = ipcv.map_rotation_scale(src, rotation=30, scale=[1.3, 0.8])
    elapsedTime = time.clock() - startTime
    print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

    startTime = time.clock()
    dst = cv2.remap(src, map1, map2, cv2.INTER_NEAREST)
    #   dst = ipcv.remap(src, map1, map2, ipcv.INTER_NEAREST)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (remapping) = {0} [s]'.format(elapsedTime))

    srcName = 'Source (' + filename + ')'
    cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(srcName, src)

    dstName = 'Destination (' + filename + ')'
    cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(dstName, dst)

    ipcv.flush()

"""
PYTHON METHOD DEFINITION
"""
def map_gcp(src, map, srcX, srcY, mapX, mapY, order=1):
    """
    Function to distort map image to fall on the coordinates of the source

    Args:
        src (array): source image
        map (array): image to distort
        srcX (array): list of x values for a set of points on the source image
        srcY (array): list of y values for a set of points on the source image
        mapX (array): list of x values for a set of points on the map image
        mapY (array): list of y values for a set of points on the map image
        order (optional[int]): order of polynomial to do mapping

    Returns:
        An image of the map with the points placed in the coordinates of the src
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
    imgFilename = home + os.path.sep + \
               'src/python/examples/data/registration/image.tif'
    mapFilename = home + os.path.sep + \
               'src/python/examples/data/registration/map.tif'
    gcpFilename = home + os.path.sep + \
               'src/python/examples/data/registration/gcp.dat'
    src = cv2.imread(srcFilename)
    map = cv2.imread(mapFilename)

    srcX = []
    srcY = []
    mapX = []
    mapY = []
    linesRead = 0
    f = open(gcpFilename, 'r')
    for line in f:
    linesRead += 1
    if linesRead > 2:
       data = line.rstrip().split()
       srcX.append(float(data[0]))
       srcY.append(float(data[1]))
       mapX.append(float(data[2]))
       mapY.append(float(data[3]))
    f.close()

    startTime = time.clock()
    map1, map2 = ipcv.map_gcp(src, map, srcX, srcY, mapX, mapY, order=2)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

    startTime = time.clock()
    dst = cv2.remap(src, map1, map2, cv2.INTER_NEAREST)
    #   dst = ipcv.remap(src, map1, map2, ipcv.INTER_NEAREST)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))

    srcName = 'Source (' + srcFilename + ')'
    cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(srcName, src)

    mapName = 'Map (' + mapFilename + ')'
    cv2.namedWindow(mapName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(mapName, map)

    dstName = 'Warped (' + mapFilename + ')'
    cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(dstName, dst)

    ipcv.flush()
