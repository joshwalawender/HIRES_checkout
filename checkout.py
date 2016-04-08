#!/usr/env/python

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging

import numpy as np
from astropy import units as u
from ccdproc import CCDData, Combiner, subtract_bias

##-------------------------------------------------------------------------
## HIRESinstrument Object
##-------------------------------------------------------------------------
class HIRESinstrument(object):
    def __init__(self):
        self.servername = 'hiresserver.keck.hawaii.edu'
    
    def take_biases(self, nbiases=8, fake=False):
        '''Script to use KTL and keywords to take n bias frames and record the
        resulting filenames to a list.
        '''
        if fake:
            bias_files = ['bias{:02d}.fits'.format(i+1)\
                          for i in np.arange(nbiases)]
            return bias_files
        bias_files = []
        ## Take bias frames here
        return bias_files

    def take_darks(self, ndarks=3, exptimes=[1, 10, 30, 100], fake=False):
        '''Script to use KTL and keywords to take n dark frames at each of the
        specified exposure times and record the resulting filenames to a list.
        '''
        if fake:
            for exptime in exptimes:
                dark_files = ['dark_exp{:03d}_{:02d}.fits'.format(exptime, i+1)\
                              for i in np.arange(ndarks)]
            return dark_files
        dark_files = []
        ## Take bias frames here
        return dark_files


##-------------------------------------------------------------------------
## Main Program
##-------------------------------------------------------------------------
def main():

    ##-------------------------------------------------------------------------
    ## Parse Command Line Arguments
    ##-------------------------------------------------------------------------
    ## create a parser object for understanding command-line arguments
    parser = argparse.ArgumentParser(
             description="Program description.")
    ## add flags
    parser.add_argument("-v", "--verbose",
        action="store_true", dest="verbose",
        default=False, help="Be verbose! (default = False)")
    parser.add_argument("--fake",
        action="store_true", dest="fake",
        default=False, help="Use test data, do not take new images.")
    ## add arguments
#     parser.add_argument("--input",
#         type=str, dest="input",
#         help="The input.")
    args = parser.parse_args()

    ##-------------------------------------------------------------------------
    ## Create logger object
    ##-------------------------------------------------------------------------
    logger = logging.getLogger('MyLogger')
    logger.setLevel(logging.DEBUG)
    ## Set up console output
    LogConsoleHandler = logging.StreamHandler()
    if args.verbose:
        LogConsoleHandler.setLevel(logging.DEBUG)
    else:
        LogConsoleHandler.setLevel(logging.INFO)
    LogFormat = logging.Formatter('%(asctime)23s %(levelname)8s: %(message)s')
    LogConsoleHandler.setFormatter(LogFormat)
    logger.addHandler(LogConsoleHandler)
    ## Set up file output
#     LogFileName = None
#     LogFileHandler = logging.FileHandler(LogFileName)
#     LogFileHandler.setLevel(logging.DEBUG)
#     LogFileHandler.setFormatter(LogFormat)
#     logger.addHandler(LogFileHandler)

    hires = HIRESinstrument()

    ##-------------------------------------------------------------------------
    ## Determine Read Noise
    ##-------------------------------------------------------------------------
    nbiases = 8
    bias_files = hires.take_biases(nbiases=nbiases, fake=args.fake)
    print(bias_files)
    ## Create master bias from first n-1 frames
    bias_data = CCDData.read(bias_files[0])
    master_bias_data = []
    for i,bias_file in enumerate(bias_files):
        if i != 0:
            master_bias_data.append(CCDData.read(bias_file))
    combiner = Combiner(master_bias_data)
    master_bias = combiner.median_combine() # combiner.average_combine
    bias_diff = subtract_bias(bias_data, master_bias)
    RN = np.std(bias_diff.data) * 1./(1. + 1./np.sqrt(nbiases-1))



if __name__ == '__main__':
    main()
