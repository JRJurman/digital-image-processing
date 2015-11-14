"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""

import numpy as np
import numerical

def idft2(f, scale=True):
    """
    Function to evaluate the inverse of the dft for a 2D array.

    Args:
        f (2d-array): array to calculate the 2 dimensional dft of
        scale (optional[boolean]): scales the result by dividing it by the
                                    number of array elements

    Returns:
        2 dimensional array of inverse fourier transformation
    """

    # for every column, evaluate the dft
    columns = f.swapaxes(0,1)
    Columns = []
    for col in range(columns.shape[0]):
        Columns.append(numerical.idft(columns[col], False))

    newSpace = np.array(Columns)

    # for every row in the new space, evaluate the dft
    # swap the axes again to set it back to normal
    rows = newSpace.swapaxes(0,1)
    Rows = []
    for row in range(rows.shape[0]):
        Rows.append(numerical.idft(rows[row], False))

    finalSpace = np.array(Rows)

    M = f.shape[0]*f.shape[1]
    scaleFactor = 1
    if (scale):
        scaleFactor = (1/M)

    return finalSpace*scaleFactor

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':
    import numerical
    import numpy
    import time

    M = 2**5
    N = 2**5
    F = numpy.zeros((M,N), dtype=numpy.complex128)
    F[0,0] = 1

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        f = numerical.idft2(F)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point iDFT2)'
    print(string.format((time.clock() - startTime)/repeats, M, N))

    startTime = time.clock()
    for repeat in range(repeats):
        f = numpy.fft.ifft2(F)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point iFFT2)'
    print(string.format((time.clock() - startTime)/repeats, M, N))
