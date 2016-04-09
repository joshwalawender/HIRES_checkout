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

## Suppress astropy log
from astropy import log
log.setLevel('ERROR')
# log.disable_warnings_logging()

##-------------------------------------------------------------------------
## HIRES Instrument Object
##-------------------------------------------------------------------------
class HIRESinstrument(object):
    def __init__(self, logger=None):
        self.servername = 'hiresserver.keck.hawaii.edu'
        self.logger = logger
    
    def take_biases(self, nbiases=10, fake=False):
        '''Script to use KTL and keywords to take n bias frames and record the
        resulting filenames to a list.
        '''
        if self.logger: self.logger.info('Taking {} bias frames'.format(nbiases))
        if fake:
            ## NOTE: this assumes nbiases=10 at the moment
            nbiases = 10
            if self.logger: self.logger.info('  Fake keyword invoked.  No data will be taken.')
            if self.logger: self.logger.info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            bias_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(1,nbiases+1,1)]
            for bias_file in bias_files:
                if self.logger: self.logger.debug('  Bias: {}'.format(bias_file))
        else:
            ## Take bias frames here
            bias_files = []

        if self.logger: self.logger.info('  Obtained {} bias files'.format(len(bias_files)))
        return bias_files

    def take_darks(self, ndarks=5, exptimes=[1, 60, 600], fake=False):
        '''Script to use KTL and keywords to take n dark frames at the
        specified exposure times and record the resulting filenames to a list.
        '''
        if self.logger: self.logger.info('Taking dark frames')
        if fake:
            ## NOTE: this assumes ndarks=5 and exptimes=[1, 60, 600]
            ndarks=5
            exptimes=[1, 60, 600]
            if self.logger: self.logger.info('  Fake keyword invoked.  No data will be taken.')
            if self.logger: self.logger.info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            start_index = 11
            dark_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(start_index,start_index+ndarks*(len(exptimes)),1)]
            for dark_file in dark_files:
                if self.logger: self.logger.debug('  {}'.format(dark_file))
            return dark_files
        dark_files = []
        ## Take bias frames here
        return dark_files


##-------------------------------------------------------------------------
## HIRES Image Object
##-------------------------------------------------------------------------
class HIRESimage(object):
    def __init__(self, fits_file=None, logger=None):
        self.logger = logger
        self.b = None
        self.g = None
        self.r = None
        self.b_mean = None
        self.g_mean = None
        self.r_mean = None
        self.b_median = None
        self.g_median = None
        self.r_median = None
        self.b_stddev = None
        self.g_stddev = None
        self.r_stddev = None
        if fits_file:
            self.read_fits(fits_file)

    def read_fits(self, fits_file):
        if self.logger: self.logger.debug('Reading {}'.format(fits_file))
        self.b = CCDData.read(fits_file, unit=u.adu, hdu=1)
        self.g = CCDData.read(fits_file, unit=u.adu, hdu=2)
        self.r = CCDData.read(fits_file, unit=u.adu, hdu=3)

    def load_from_CCDData(self, b, g, r):
        if self.logger: self.logger.debug('Loading data from CCDData objects')
        self.b = b
        self.g = g
        self.r = r

    def calculate_stats(self, sigma=9, iters=0):
        assert self.b
        assert self.g
        assert self.r
        self.b_mean, self.b_median, self.b_stddev = stats.sigma_clipped_stats(\
                                                    self.b.data,\
                                                    sigma=sigma, iters=iters)
        self.g_mean, self.g_median, self.g_stddev = stats.sigma_clipped_stats(\
                                                    self.g.data,\
                                                    sigma=sigma, iters=iters)
        self.r_mean, self.r_median, self.r_stddev = stats.sigma_clipped_stats(\
                                                    self.r.data,\
                                                    sigma=sigma, iters=iters)


    def subtract_bias(self, image):
        biassub = HIRESimage(logger=self.logger)
        biassub.load_from_CCDData(subtract_bias(self.b, image.b),\
                                  subtract_bias(self.g, image.g),\
                                  subtract_bias(self.r, image.r))
        return biassub






##-------------------------------------------------------------------------
## HIRES Image Functions
##-------------------------------------------------------------------------
def HIREScombine(image_list, combine='median', logger=None):
    
    image_list_b = [im.b for im in image_list]
    image_list_g = [im.g for im in image_list]
    image_list_r = [im.r for im in image_list]
    
    combiner_b = Combiner(image_list_b)
    combiner_g = Combiner(image_list_g)
    combiner_r = Combiner(image_list_r)
    
    if combine.lower() in ['median', 'med']:
        combined_b_ccdobj = combiner_b.median_combine()
        combined_g_ccdobj = combiner_g.median_combine()
        combined_r_ccdobj = combiner_r.median_combine()
    elif combine.lower() in ['average', 'avg']:
        combined_b_ccdobj = combiner_b.average_combine()
        combined_g_ccdobj = combiner_g.average_combine()
        combined_r_ccdobj = combiner_r.average_combine()
    else:
        combined_b_ccdobj = combiner_b.average_combine()
        combined_g_ccdobj = combiner_g.average_combine()
        combined_r_ccdobj = combiner_r.average_combine()

    combined = HIRESimage(logger=logger)
    combined.load_from_CCDData(combined_b_ccdobj,\
                               combined_g_ccdobj,\
                               combined_r_ccdobj)
    return combined


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
    clipping_sigma = 9
    clipping_iters = 1
    nbiases = 10
    bias_files = hires.take_biases(nbiases=nbiases, fake=args.fake)
    nbiases = len(bias_files)
    ## First bias frame is the reference
    bias = HIRESimage(fits_file=bias_files[0], logger=logger)
    bias.calculate_stats(sigma=clipping_sigma, iters=clipping_iters)
    logger.debug('  Bias B (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias.b_mean, bias.b_median, bias.b_stddev))
    logger.debug('  Bias G (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias.g_mean, bias.g_median, bias.g_stddev))
    logger.debug('  Bias R (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias.r_mean, bias.r_median, bias.r_stddev))
    
    ## Create master bias from next n-1 frames
    logger.info('Reading bias frames for creating master bias')
    image_list = []
    for i,bias_file in enumerate(bias_files):
        if i != 0:
            image_list.append(HIRESimage(fits_file=bias_file, logger=logger))
    logger.info('Combining {} files to make master bias'.format(len(image_list)))
    master_bias = HIREScombine(image_list, combine='median', logger=logger)
    master_bias.calculate_stats(sigma=clipping_sigma, iters=clipping_iters)
    logger.debug('  Master Bias B (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 master_bias.b_mean, master_bias.b_median, master_bias.b_stddev))
    logger.debug('  Master Bias G (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 master_bias.g_mean, master_bias.g_median, master_bias.g_stddev))
    logger.debug('  Master Bias R (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 master_bias.r_mean, master_bias.r_median, master_bias.r_stddev))

    ## Create difference between a single bias frame and the master bias
    logger.info('Creating bias difference frames')
    bias_diff = bias.subtract_bias(master_bias)
    bias_diff.calculate_stats(sigma=clipping_sigma, iters=clipping_iters)
    logger.debug('  Bias Difference B (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias_diff.b_mean, bias_diff.b_median, bias_diff.b_stddev))
    logger.debug('  Bias Difference G (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias_diff.g_mean, bias_diff.g_median, bias_diff.g_stddev))
    logger.debug('  Bias Difference R (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                 bias_diff.r_mean, bias_diff.r_median, bias_diff.r_stddev))

    ## Calculate read noise from the difference frame's stddev
    RN_b = bias_diff.b_stddev / np.sqrt(1.+1./(nbiases-1))
    RN_g = bias_diff.g_stddev / np.sqrt(1.+1./(nbiases-1))
    RN_r = bias_diff.r_stddev / np.sqrt(1.+1./(nbiases-1))
    logger.info('Read Noise (blue) = {:.2f} ADU'.format(RN_b))
    logger.info('Read Noise (green) = {:.2f} ADU'.format(RN_g))
    logger.info('Read Noise (red) = {:.2f} ADU'.format(RN_r))

    ##-------------------------------------------------------------------------
    ## Determine Dark Current
    ##-------------------------------------------------------------------------
    ndarks = 5
    exptimes=[1, 60, 600]
    dark_files = hires.take_darks(ndarks=ndarks, exptimes=exptimes, fake=args.fake)
    dark_table = Table(names=('filename', 'exptime', 'mean', 'median', 'stddev'),\
                       dtype=('a64', 'f4', 'f4', 'f4', 'f4'))
#     for dark_file in dark_files:
#         dark_ccdobj = CCDData.read(dark_file, unit=u.adu)
#         exptime = float(dark_ccdobj.header['EXPTIME'])
#         mean, median, stddev = stats.sigma_clipped_stats(dark_ccdobj.data, sigma=5, iters=3)
#         dark_table.add_row([dark_file, exptime, mean, median, stddev])
#     print(dark_table)



if __name__ == '__main__':
    main()
