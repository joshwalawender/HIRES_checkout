#!kpython

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging
from glob import glob

import numpy as np
from astropy import units as u
# from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astropy import stats

import ccdproc

## Suppress astropy log
from astropy import log
log.setLevel('ERROR')
# log.disable_warnings_logging()

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
import matplotlib.pyplot as plt

# from instruments import HIRES



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
        default=True, help="Use test data, do not take new images.")
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
    LogFormat = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s',
                                  datefmt='%Y%m%d %H:%M:%S')
    LogConsoleHandler.setFormatter(LogFormat)
    logger.addHandler(LogConsoleHandler)
    ## Set up file output
#     LogFileName = None
#     LogFileHandler = logging.FileHandler(LogFileName)
#     LogFileHandler.setLevel(logging.DEBUG)
#     LogFileHandler.setFormatter(LogFormat)
#     logger.addHandler(LogFileHandler)

#     hires = HIRES(logger=logger)

    ##-------------------------------------------------------------------------
    ## Determine Read Noise
    ##-------------------------------------------------------------------------
    if args.fake:
        bias_files = glob('/home/jwalawender/HIRES_Test_Data/*.fits')[0:10]
    else:
        pass
#         nbiases = 3
#         bias_files = hires.take_bias(n=nbiases)

    nbiases = len(bias_files)

    ## Determine Bias Stats for each Chip
    clipping_sigma = 5
    clipping_iters = 1
    plt.figure(figsize=(9,15), dpi=72)
    binsize = 1.0
    for chip in [1,2,3]:
        logger.info('Analyzing Chip {:d}'.format(chip))
        ax = plt.subplot(3,1,chip)
        biases = []
        for bias_file in bias_files:
            logger.debug('  Reading {}'.format(bias_file))
            if bias_file == bias_files[0]:
                bias = ccdproc.fits_ccddata_reader(bias_file, unit='adu', hdu=chip)
                mean, median, stddev = stats.sigma_clipped_stats(bias.data,
                                             sigma=clipping_sigma,
                                             iters=clipping_iters) * u.adu
                logger.debug('  Bias ext={:d} (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                             chip, mean, median, stddev))
            else:
                biases.append(ccdproc.fits_ccddata_reader(bias_file, unit='adu', hdu=chip))
        logger.debug('  Making master bias')
        master_bias = ccdproc.combine(biases, combine='median')
        mean, median, stddev = stats.sigma_clipped_stats(master_bias.data,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        logger.debug('  Master Bias ext={:d} (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                     chip, mean, median, stddev))

        diff = bias.subtract(master_bias)
        mean, median, stddev = stats.sigma_clipped_stats(diff.data,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        logger.debug('  Bias Difference ext={:d} (mean, median, stddev) = {:.2f}, {:.2f}, {:.2f}'.format(\
                     chip, mean, median, stddev))
        RN = stddev / np.sqrt(1.+1./(nbiases-1))
        logger.info('Read Noise (ext={:d}) is {:.2f}'.format(chip, RN))

        ##---------------------------------------------------------------------
        ## Plot Bias Frame Histograms
        ##---------------------------------------------------------------------
        logger.debug('  Generating histogram of bias difference image values')
        bmin = np.floor(min(diff.data.ravel())/binsize)*binsize - binsize/2.
        bmax = np.ceil(max(diff.data.ravel())/binsize)*binsize + binsize/2.
        bins = np.arange(bmin,bmax,binsize)
        hist, bins = np.histogram(diff.data.ravel(), bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        gaussian = models.Gaussian1D(amplitude=max(hist),\
                                     mean=mean,\
                                     stddev=RN)
        gaussian_plot = [gaussian(x) for x in centers]
        plt.bar(centers, hist,\
                align='center', width=0.7*binsize, log=True, color='b', alpha=0.5,\
                label='Blue CCD Pixel Count Histogram')
        plt.plot(centers, gaussian_plot, 'b-', alpha=0.8,\
                 label='Gaussian with sigma = {:.2f}'.format(RN))
        plt.plot([mean.value, mean.value], [1, 2*max(hist)], label='Mean Pixel Value')
        plt.xlim(np.floor(mean.value-7.*RN.value), np.ceil(mean.value+7.*RN.value))
        plt.ylim(1, 2*max(hist))
        ax.set_xlabel('Counts (ADU)', fontsize=10)
        ax.set_ylabel('Number of Pixels', fontsize=10)
        ax.grid()
        ax.legend(loc='best', fontsize=10)

    plotfilename = 'BiasHistogram.png'
    logger.info('Saving: {}'.format(plotfilename))
    plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
    plt.close()
    logger.info('Done.')



    sys.exit(0)



    ##-------------------------------------------------------------------------
    ## Determine Dark Current
    ##-------------------------------------------------------------------------
    ndarks = 5
    exptimes=[1, 60, 600]
    clipping_sigma = 5
    clipping_iters = 3
    dark_files = []
    for exptime in exptimes:
        new_files = hires.take_darks(n=ndarks, exptimes=exptimes)
        dark_files.extend(new_files)
    dark_table = Table(names=('filename', 'exptime', 'chip', 'mean', 'median', 'stddev'),\
                       dtype=('a64', 'f4', 'a4', 'f4', 'f4', 'f4'))
    logger.info('Analyzing bias subtracted dark frames to measure dark current.')
    logger.info('  Determining image statistics using sigma clipping algorithm.')
    logger.info('    sigma={:d}, iters={:d}'.format(clipping_sigma, clipping_iters))
    for dark_file in dark_files:
        dark = HIRESimage(fits_file=dark_file, logger=logger)
        exptime = float(dark.header['EXPTIME'])
        dark_diff = dark.subtract_bias(master_bias)
        dark_diff.calculate_stats(sigma=clipping_sigma, iters=clipping_iters)
        dark_table.add_row([dark_file, exptime, 'B', dark_diff.b_mean,\
                            dark_diff.b_median, dark_diff.b_stddev])
        dark_table.add_row([dark_file, exptime, 'G', dark_diff.g_mean,\
                            dark_diff.g_median, dark_diff.g_stddev])
        dark_table.add_row([dark_file, exptime, 'R', dark_diff.r_mean,\
                            dark_diff.r_median, dark_diff.r_stddev])
    dark_table.write('dark_table.txt', format='ascii.csv')

    dark_table = Table.read('dark_table.txt', format='ascii.csv')

    logger.info('  Fitting line to levels as a function of exposure time')
    line = models.Linear1D(intercept=0, slope=0)
    line.intercept.fixed = True
    fitter = fitting.LinearLSQFitter()
    dc_fit_B = fitter(line, dark_table[dark_table['chip'] == 'B']['exptime'],\
                            dark_table[dark_table['chip'] == 'B']['mean'])
    dc_fit_G = fitter(line, dark_table[dark_table['chip'] == 'G']['exptime'],\
                            dark_table[dark_table['chip'] == 'G']['mean'])
    dc_fit_R = fitter(line, dark_table[dark_table['chip'] == 'R']['exptime'],\
                            dark_table[dark_table['chip'] == 'R']['mean'])
    logger.info('Dark Current (blue) = {:.2f} ADU/600s'.format(dc_fit_B.slope.value*600.))
    logger.info('Dark Current (green) = {:.2f} ADU/600s'.format(dc_fit_G.slope.value*600.))
    logger.info('Dark Current (red) = {:.2f} ADU/600s'.format(dc_fit_R.slope.value*600.))


    ##-------------------------------------------------------------------------
    ## Plot Dark Frame Levels
    ##-------------------------------------------------------------------------
    logger.info('Generating figure with dark current fits')
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(dark_table[dark_table['chip'] == 'B']['exptime'],\
            dark_table[dark_table['chip'] == 'B']['mean'],\
            'bo',\
            label='mean count level in ADU (B)',\
            alpha=1.0)
    ax.plot([0, 1000],\
            [dc_fit_B(0), dc_fit_B(1000)],\
            'b-',\
            label='dark current (B) = {:.2f} ADU/600s'.format(dc_fit_B.slope.value*600.),\
            alpha=0.3)
    ax.plot(dark_table[dark_table['chip'] == 'G']['exptime'],\
            dark_table[dark_table['chip'] == 'G']['mean'],\
            'go',\
            label='mean count level in ADU (G)',\
            alpha=1.0)
    ax.plot([0, 1000],\
            [dc_fit_G(0), dc_fit_G(1000)],\
            'g-',\
            label='dark current (G) = {:.2f} ADU/600s'.format(dc_fit_G.slope.value*600.),\
            alpha=0.3)
    ax.plot(dark_table[dark_table['chip'] == 'R']['exptime'],\
            dark_table[dark_table['chip'] == 'R']['mean'],\
            'ro',\
            label='mean count level in ADU (R)',\
            alpha=1.0)
    ax.plot([0, 1000],\
            [dc_fit_R(0), dc_fit_R(1000)],\
            'r-',\
            label='dark current (R) = {:.2f} ADU/600s'.format(dc_fit_R.slope.value*600.),\
            alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.xlim(0, 1.1*max(dark_table['exptime']))
    plt.ylim(np.floor(min(dark_table['mean'])), np.ceil(max(dark_table['mean'])))
    ax.set_xlabel('Exposure Time (s)')
    ax.set_ylabel('Dark Level (ADU)')
    ax.grid()
    ax.legend(loc='best', fontsize=10)
    ax.set_title('Dark Current'\
                 '\n(Mean computed using sigma clipping: sigma={:d}, '\
                 'iterations={:d})'.format(\
                 clipping_sigma, clipping_iters), fontsize=10)
    ax.grid()

    plotfilename = 'DarkCurrent.png'
    logger.info('  Saving: {}'.format(plotfilename))
    plt.savefig(plotfilename, dpi=100)
    plt.close()
    logger.info('  Done.')



if __name__ == '__main__':
    main()
