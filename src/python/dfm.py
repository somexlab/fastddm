import argparse

def isf(img_seq,**kwargs):
    """Compute image structure function from image sequence

    :param img_seq: The image sequence
    :type img_seq: _type_
    :param backend: Compute with python or C++
    :type backend: str
    :param fft_mode: Use fft (Wiener-Khinchin theorem) to compute the structure function.
    :type fft_mode: bool
    :param fft2_opt: Optimization mode for fft2. Possible options are: 'none', 'powerof2', 'power'.
    :type fft2_opt: str
    :param fft1_opt: Optimization mode for fft1 (only used if fft_mode is True). Possible options are: 'none', 'powerof2', 'power'.
    :type fft1_opt: str
    :param lags: List of lags to be analyzed.
    :type lags: List[int]

    :return: ISF object
    :rtype: ImageStructureFunction
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    - input can be either a filename (the multipage tiff file or a txt file with paths) or a folder containing tiff images which we need to sort.
    - fft optimizations as flags+string (ex. --fft2_opt=powerof2)
    - config file (TBD)
    - output directory (we decide file names)
    - backend option ()
    - lags as txt file or mode-string (eg fibonacci, logspaced, all[default])
    - compute mode (with or without fft)
    - test mode (TBD)
    """