"""
PYTHON METHOD DEFINITION
by Jesse Jurman (jrj2703)
"""
import numerical
def idft(f, scale=True):
    """
    Function to evaluate the inverse of the dft for an array.

    Args:
        f (ndarray): array to calculate the dft of
        scale (optional[boolean]): scales the result by dividing it by the
                                    number of array elements

    Returns:
        Array of fourier transformation
    """

    def swap(f):
        """
        Function to swap the imaginary and real components of an imaginary array
        """
        return f.real*(0+1j) + f.imag

    M = f.shape[0]
    scaleFactor = 1
    if (scale):
        scaleFactor = (1/M)
    transform = swap(numerical.dft(swap(f)))/scale

    return transform
"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':
    import numerical
    import numpy
    import time

    N = 2**12
    F = numpy.zeros(N, dtype=numpy.complex128)
    F[0] = 1

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        f = numerical.idft(F)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point iDFT)'
    print(string.format((time.clock() - startTime)/repeats, len(F)))

    startTime = time.clock()
    for repeat in range(repeats):
        f = numpy.fft.ifft(F)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point iFFT)'
    print(string.format((time.clock() - startTime)/repeats, len(F)))
