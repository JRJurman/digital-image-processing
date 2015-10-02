"""
IMPORTS
"""
import numpy as np
import ipcv
import cv2

"""
PYTHON METHOD DEFINITION
"""
def map_gcp(src, map, srcX, srcY, mapX, mapY, order=1):
    """
    Function to distort map image to fall on the coordinates of the source

    Args:
        src (array): source (image to distort)
        map (array): image to register with source
        srcX (array): list of x values for a set of points on the source image
        srcY (array): list of y values for a set of points on the source image
        mapX (array): list of x values for a set of points on the map image
        mapY (array): list of y values for a set of points on the map image
        order (optional[int]): order of polynomial to do mapping

    Returns:
        two maps, of x values and y values

    Raises:
        ValueError if order is above 2
    """

    if (order > 2):
        raise ValueError("order parameter should be less than 2")

    # get the exponent terms for x and y
    nterms = (order+1)**2
    x = np.arange(nterms)
    y = [[0],[1]]

    mesh, grab = np.meshgrid(x, y)

    xExp = np.floor( mesh[0] % (order + 1) )
    yExp = np.floor(mesh[1] / (order+1))

    # build design matrix from map points
    X = np.zeros((len(mapX), nterms))

    for ind in range(len(mapX)):
        for term in range(nterms):
            X[ind, term] = (mapX[ind]**xExp[term])*(mapY[ind]**yExp[term])

    # build coefficients for C
    # now we need both src and map
    Y = np.asmatrix([srcX, srcY]).T
    Xm = np.asmatrix(X)

    # from the notes
    Xsq = (Xm.T * Xm)
    C = Xsq.I * Xm.T * Y
    # this will be a0, a1, a2, etc...

    # building the final maps before we return
    xs, ys = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))

    # doing our transform
    Xp, Yp = 0, 0
    for term in range(nterms):
        Xp += C[term, 0] * (xs**xExp[term]) * (ys**yExp[term])
        Yp += C[term, 1] * (xs**xExp[term]) * (ys**yExp[term])

    Xp = Xp.astype('float32')
    Yp = Yp.astype('float32')

    return Xp, Yp

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time

    home = os.path.expanduser('~')
    srcFilename = home + os.path.sep + \
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
