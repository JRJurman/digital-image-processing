"""
IMPORTS
"""
import numpy as np
import ipcv
import cv2

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
        two maps, of x values and y values
    """

    deg = np.radians(rotation)

    # rotation matrix (based on slides)
    rotateMatrix = np.array([
        np.cos(deg),
        np.sin(deg),
        -np.sin(deg),
        np.cos(deg)
    ])
    rotateMatrix = rotateMatrix.reshape(2,2)

    # scale matrix (based on notes)
    scaleMatrix = np.array([
        scale[0], 0,
        0, scale[1]
    ])
    scaleMatrix = scaleMatrix.reshape(2,2)

    # transformations
    transformMatrix = np.dot(rotateMatrix, scaleMatrix)

    # get the size of the Destination Map
    corners = np.array([
        [-src.shape[0]/2, src.shape[1]/2],
        [src.shape[0]/2, src.shape[1]/2],
        [-src.shape[0]/2, -src.shape[1]/2],
        [src.shape[0]/2, -src.shape[1]/2]
    ])

    # set up min and max values to compare against with
    # the transformed corners
    minX, minY = float('inf'), float('inf')
    maxX, maxY = float('-inf'), float('-inf')

    for xyPair in corners:
        x, y = np.dot(transformMatrix, xyPair)
        minX = min(minX, x)
        minY = min(minY, y)
        maxX = max(maxX, x)
        maxY = max(maxY, y)

    xwidth = maxX - minX
    ywidth = maxY - minY

    xs = np.zeros((ywidth, xwidth))
    ys = np.zeros((ywidth, xwidth))

    """
    xs = np.meshgrid(xwidth)
    ys = np.meshgrid(ywidth)

    # offset values to center rotation
    xs = xs - (xwidth/2)
    ys = (ywidth/2) - ys

    tm = np.asmatrix(transformMatrix).I
    xs, ys = tm*(np.array([xs, ys]).reshape(2,1))

    xs = xs + (src.shape[1]/2)
    ys = (src.shape[0]/2) - ys
    """
    
    for row in range(xs.shape[0]):
        for col in range(xs.shape[1]):

            # offset the values to center the rotation
            offX = col - (xs.shape[1]/2)
            offY = (xs.shape[0]/2) - row

            offs = np.array([offX, offY])

            # transform the points
            tm = np.asmatrix(transformMatrix).I
            offs = np.asmatrix(offs.reshape(2,1))
            xp, yp = tm*offs

            # un-offset them, and place them in the final maps
            unoffX = (xp + (src.shape[1]/2))
            unoffY = ((src.shape[0]/2) - yp)

            xs[row,col] = unoffX
            ys[row,col] = unoffY

    xs = xs.astype('float32')
    ys = ys.astype('float32')
    return (xs, ys)


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
    #map1, map2 = ipcv.map_rotation_scale(src, rotation=0, scale=[2.0, 2.0])
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
