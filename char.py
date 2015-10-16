"""
PYTHON METHOD DEFINITION
"""
def character_recognition(src, templates, threshold, filterType='spatial'):
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
            defaults to 'spatial'

    Returns:
        a string (or list of characters)
        a histogram (size of the array templates)

    Raises:
        ValueError: image shape is not 2 (gray-scale) or 3 (color)
    """

"""
PYTHON TEST HARNESS
"""
if __name__ == '__main__':

    import cv2
    import fnmatch
    import numpy
    import os
    import os.path

    home = os.path.expanduser('~')
    baseDirectory = home + os.path.sep + 'src/python/examples/data'
    baseDirectory += os.path.sep + 'character_recognition'

    documentFilename = baseDirectory + '/notAntiAliased/text.tif'
    documentFilename = baseDirectory + '/notAntiAliased/alphabet.tif'
    charactersDirectory = baseDirectory + '/notAntiAliased/characters'

    document = cv2.imread(documentFilename, cv2.IMREAD_UNCHANGED)

    characterImages = []
    for root, dirnames, filenames in os.walk(charactersDirectory):
        for filename in filenames:
            currentCharacter = cv2.imread(root + '/' + filename,
                                                        cv2.IMREAD_UNCHANGED)
            characterImages.append(currentCharacter)
    characterImages = numpy.asarray(characterImages)

    # Define the filter threshold
    """
    threshold = ...
    """

    text, histogram = character_recognition(document, characterImages, threshold, filterType='spatial')

    """
    # Display the results to the user
    .
    .
    .
    """

    text, histogram = character_recognition(document, characterImages, threshold, filterType='matched')

    """
    # Display the results to the user
    .
    .
    .
    """
