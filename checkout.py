#!/usr/env/python

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging

import numpy as np
from astropy import units as u
from astropy.table import Table, Column
from astropy import stats
from ccdproc import CCDData, Combiner, subtract_bias

##-------------------------------------------------------------------------
## HIRESinstrument Object
##-------------------------------------------------------------------------
class HIRESinstrument(object):
    def __init__(self, logger=None):
        self.servername = 'hiresserver.keck.hawaii.edu'
        self.logger = logger
    
    def take_biases(self, nbiases=10, fake=False):
        '''Script to use KTL and keywords to take n bias frames and record the
        resulting filenames to a list.
        '''
        self.logger.info('Taking {} bias frames'.format(nbiases))
        if fake:
            ## NOTE: this assumes nbiases=10 at the moment
            nbiases = 10
            self.logger.info('  Fake keyword invoked.  No data will be taken.')
            self.logger.info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            bias_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(1,11,1)]
            for bias_file in bias_files:
                self.logger.debug('  Bias: {}'.format(bias_file))
            return bias_files
        bias_files = []
        ## Take bias frames here
        return bias_files

    def take_darks(self, ndarks=5, exptime=1, fake=False):
        '''Script to use KTL and keywords to take n dark frames at the
        specified exposure time and record the resulting filenames to a list.
        '''
        self.logger.info('Taking {} {}s dark frames'.format(ndarks, exptime))
        if fake:
            ## NOTE: this assumes ndarks=5 at the moment
            ndarks=5
            self.logger.info('  Fake keyword invoked.  No data will be taken.')
            self.logger.info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            if exptime == 1:
                start_index = 11
            elif exptime == 60:
                start_index = 16
            elif exptime == 600:
                start_index = 21
            else:
                start_index = 21
                logger.warning('No darks of this length in the fake data set')
            dark_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(start_index,start_index+ndarks,1)]
            for dark_file in dark_files:
                self.logger.debug('  {}'.format(dark_file))
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

    hires = HIRESinstrument(logger=logger)

    ##-------------------------------------------------------------------------
    ## Determine Read Noise
    ##-------------------------------------------------------------------------
    nbiases = 10
    bias_files = hires.take_biases(nbiases=nbiases, fake=args.fake)
    ## First bias frame is the reference
    logger.info('Reading first bias frame')
    bias_b_ccdobj = CCDData.read(bias_files[0], unit=u.adu, hdu=0)
    logger.debug('  Read blue chip')
    bias_g_ccdobj = CCDData.read(bias_files[0], unit=u.adu, hdu=1)
    logger.debug('  Read green chip')
    bias_r_ccdobj = CCDData.read(bias_files[0], unit=u.adu, hdu=2)
    logger.debug('  Read red chip')

    ## Create master bias from next n-1 frames
    master_bias_data_b = []
    master_bias_data_g = []
    master_bias_data_r = []
    for i,bias_file in enumerate(bias_files):
        if i != 0:
            logger.info('Reading bias frame {}'.format(i+1))
            master_bias_data_b.append(CCDData.read(bias_file, unit=u.adu, hdu=0))
            master_bias_data_g.append(CCDData.read(bias_file, unit=u.adu, hdu=1))
            master_bias_data_r.append(CCDData.read(bias_file, unit=u.adu, hdu=2))
    logger.info('Combining frames into master biases for B, G, and R chips')
    combiner_b = Combiner(master_bias_data_b)
    combiner_g = Combiner(master_bias_data_g)
    combiner_r = Combiner(master_bias_data_r)
    master_bias_b_ccdobj = combiner_b.median_combine() # combiner.average_combine()
    master_bias_g_ccdobj = combiner_g.median_combine() # combiner.average_combine()
    master_bias_r_ccdobj = combiner_r.median_combine() # combiner.average_combine()
    bias_diff_b_ccdobj = subtract_bias(bias_b_ccdobj, master_bias_b_ccdobj)
    bias_diff_g_ccdobj = subtract_bias(bias_g_ccdobj, master_bias_g_ccdobj)
    bias_diff_r_ccdobj = subtract_bias(bias_r_ccdobj, master_bias_r_ccdobj)
    # RN should be sigma / SQRT(1+Nmaster)
    # where Nmaster = nbiases-1
    # so correct at sigma / SQRT(nbiases)
    mean, median, stddev = stats.sigma_clipped_stats(bias_diff_b_ccdobj.data, sigma=9, iters=1)
    RN_b = stddev / np.sqrt(nbiases)
    mean, median, stddev = stats.sigma_clipped_stats(bias_diff_g_ccdobj.data, sigma=9, iters=1)
    RN_g = stddev / np.sqrt(nbiases)
    mean, median, stddev = stats.sigma_clipped_stats(bias_diff_r_ccdobj.data, sigma=9, iters=1)
    RN_r = stddev / np.sqrt(nbiases)
    logger.info('Read Noise (blue) = {:.2f} ADU'.format(RN_b))
    logger.info('Read Noise (green) = {:.2f} ADU'.format(RN_g))
    logger.info('Read Noise (red) = {:.2f} ADU'.format(RN_r))

    sys.exit(0)

    ##-------------------------------------------------------------------------
    ## Determine Dark Current
    ##-------------------------------------------------------------------------
    ndarks = 5
    exptimes=[1, 60, 600]
    dark_files = []
    for exptime in exptimes:
        dark_files.extend(hires.take_darks(ndarks=ndarks, exptime=exptime, fake=args.fake))
    print(dark_files)

    dark_table = Table(names=('filename', 'exptime', 'mean', 'median', 'stddev'),\
                       dtype=('a64', 'f4', 'f4', 'f4', 'f4'))
    for dark_file in dark_files:
        dark_ccdobj = CCDData.read(dark_file, unit=u.adu)
        exptime = float(dark_ccdobj.header['EXPTIME'])
        mean, median, stddev = stats.sigma_clipped_stats(dark_ccdobj.data, sigma=5, iters=3)
        dark_table.add_row([dark_file, exptime, mean, median, stddev])
    print(dark_table)



if __name__ == '__main__':
    main()
