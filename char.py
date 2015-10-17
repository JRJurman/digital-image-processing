"""
PYTHON METHOD DEFINITION
"""
import numpy as np
import cv2
import ipcv

def invertImage(src):
    """
    Function to return inverted image
     This is a separate method to handle all the conversions required

    Args:
        src(array of ints): image to convert

    Returns:
        inverted (black -> white, white -> black) image
    """
    # convert src to negatable-source, something that can have negative values
    nsrc = np.array(src, np.int32)

    # subtract the max value, and get absolute
    psrc = np.abs(nsrc - nsrc.max())

    # return the image in the original type
    return np.array(psrc, src.dtype)



def character_recognition(src, templates, threshold, filterType='spatial', verbose=False):
    """
    Function to determine the number of characters (for each character) that
    exist in an image.

    Author:
        Jesse Jurman (jrj2703)

    Args:
        src (array of ints): image to read charcters from
        templates (array of arrays): images for each character to be read in
        threshold (float): how strong the response needs to be before it is
            considered a match
        filterType (string): type of filter to run with images.
            defaults to 'spatial', can be 'match'
        verbose (bool): plot character maps

    Returns:
        a string (or list of characters)
        a histogram (size of the array templates)

    Raises:
        ValueError: filterType of not spatial or matched
    """

    isrc = invertImage(src)
    results = []
    text = ""


    if (filterType == 'spatial'):

        # for every template, build a 2D map of where the template matches
        for character in templates:

            if (verbose):
                cv2.namedWindow('isrc', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('isrc', isrc)

            if (verbose):
                cv2.namedWindow('character', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('character', character)

            # invert the character
            icharacter = invertImage(character)

            # normalize template
            icharacter = icharacter.astype(np.float64)
            icharacter /= icharacter.sum()

            if (verbose):
                cv2.namedWindow('icharacter', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('icharacter', icharacter)

            # find points using filter2D
            cloudMatch = cv2.filter2D(src, -1, icharacter)

            if (verbose):
                cv2.namedWindow('CloudMatch', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('CloudMatch', cloudMatch)

            # make single points using np.where
            pointMatch = np.where(cloudMatch >= threshold, 0, 255)
            pointMatch = np.array(pointMatch, cloudMatch.dtype)

            if (verbose):
                cv2.namedWindow('PointMatch', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('PointMatch', pointMatch)

            if (verbose):
                ipcv.flush()

            results.append((pointMatch/255).sum())
        text = ""

    elif (filterType == 'matched'):
        pass
    else:
        raise ValueError("only filter types supported are spatial and matched")

    return text, results
"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import fnmatch
    import numpy
    import os
    import os.path
    import ipcv

    home = os.path.expanduser('~')
    baseDirectory = home + os.path.sep + 'src/python/examples/data'
    baseDirectory += os.path.sep + 'character_recognition'

    documentFilename = baseDirectory + '/notAntiAliased/alphabet.tif'
    documentFilename = baseDirectory + '/notAntiAliased/text.tif'
    charactersDirectory = baseDirectory + '/notAntiAliased/characters'

    document = cv2.imread(documentFilename, cv2.IMREAD_UNCHANGED)

    characterNames = []
    characterImages = []
    for root, dirnames, filenames in os.walk(charactersDirectory):
        for filename in sorted(filenames):
            characterNames.append(chr(int(filename.split('.')[0])))
            currentCharacter = cv2.imread(root + '/' + filename,
                                                        cv2.IMREAD_UNCHANGED)
            characterImages.append(currentCharacter)
    characterImages = numpy.asarray(characterImages)

    # Define the filter threshold
    threshold = 1.0
    text, histogram = character_recognition(document, characterImages, threshold, filterType='spatial')
    for n in range(len(characterNames)):
        print(str(characterNames[n]) +": "+ str(histogram[n]))

    ipcv.plotLetters(histogram, numpy.array(characterNames))
    text, histogram = character_recognition(document, characterImages, threshold, filterType='matched')

    """
    # Display the results to the user
    .
    .
    .
    """
