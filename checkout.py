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

from hires import HIRESinstrument, HIRESimage, HIREScombine

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
