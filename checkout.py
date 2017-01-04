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
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astropy import stats
from astropy.io import fits

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

def get_mode(im):
    '''
    Return mode of image.  Assumes int values (ADU), so uses binsize of one.
    '''
    bmin = np.floor(min(im.data.ravel())) - 1./2.
    bmax = np.ceil(max(im.data.ravel())) + 1./2.
    bins = np.arange(bmin,bmax,1)
    hist, bins = np.histogram(im.data.ravel(), bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    w = np.argmax(hist)
    mode = int(centers[w])
    return mode


##-------------------------------------------------------------------------
## Sort Input Files
##-------------------------------------------------------------------------
def get_file_list(input):
    '''
    
    '''
    assert os.path.exists(input)
    
    with open(input, 'r') as FO:
        contents = FO.read()
    files = [line.strip('\n') for line in contents.split('\n') if line != '']

    bias_files = []
    dark_files = []
    flat_files = []
    for file in files:
        hdr = fits.getheader(file, 0)
        if hdr.get('OBSTYPE').strip() == 'Bias':
            bias_files.append(file)
        elif hdr.get('OBSTYPE').strip() == 'Dark':
            dark_files.append(file)
        elif hdr.get('OBSTYPE').strip() == 'IntFlat':
            flat_files.append(file)

    dict = {'bias': bias_files, 'dark': dark_files, 'flat': flat_files}

    return dict


##-------------------------------------------------------------------------
## Determine Read Noise
##-------------------------------------------------------------------------
def read_noise(bias_files, plots=False, logger=None, chips=[1,2,3]):
    '''
    '''
    nbiases = len(bias_files)
    clipping_sigma = 5
    clipping_iters = 1
    if plots:
        plt.figure(figsize=(9,15), dpi=72)
        binsize = 1.0
    master_biases = {}
    read_noise = {}
    for chip in chips:
        logger.info('Analyzing Chip {:d}'.format(chip))
        if plots:
            ax = plt.subplot(3,1,chip)
        biases = []
        for bias_file in bias_files:
            logger.debug('  Reading {}'.format(bias_file))
            if bias_file == bias_files[0]:
                bias0 = ccdproc.fits_ccddata_reader(bias_file, unit='adu', hdu=chip)
                mean, median, stddev = stats.sigma_clipped_stats(bias0.data,
                                             sigma=clipping_sigma,
                                             iters=clipping_iters) * u.adu
                mode = get_mode(bias0)
                logger.debug('  Bias[{:d}] (mean, med, mode, std) = '\
                             '{:.1f}, {:.1f}, {:d}, {:.2f}'.format(\
                             chip, mean.value, median.value, mode, stddev.value))
            else:
                biases.append(ccdproc.fits_ccddata_reader(bias_file, unit='adu', hdu=chip))
        logger.debug('  Making master bias')
        master_bias = ccdproc.combine(biases, combine='median')
        master_biases[chip] = master_bias
        mean, median, stddev = stats.sigma_clipped_stats(master_bias.data,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        mode = get_mode(master_bias)
        logger.debug('  Master Bias[{:d}] (mean, med, mode, std) = '\
                     '{:.1f}, {:.1f}, {:d}, {:.2f}'.format(\
                     chip, mean.value, median.value, mode, stddev.value))

        diff = bias0.subtract(master_bias)
        mean, median, stddev = stats.sigma_clipped_stats(diff.data,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        mode = get_mode(diff)
        logger.debug('  Bias Difference[{:d}] (mean, med, mode, std) = '\
                     '{:.1f}, {:.1f}, {:d}, {:.2f}'.format(\
                     chip, mean.value, median.value, mode, stddev.value))
        RN = stddev / np.sqrt(1.+1./(nbiases-1))
        read_noise[chip] = RN
        logger.info('For chip #{:d}:'.format(chip))
        logger.info('  Read Noise is {:.2f}'.format(RN))

        ##---------------------------------------------------------------------
        ## Plot Bias Frame Histograms
        ##---------------------------------------------------------------------
        if plots:
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
            plt.bar(centers, hist,
                    align='center', width=0.7*binsize, log=True, color='b',
                    alpha=0.5,
                    label='Blue CCD Pixel Count Histogram')
            plt.plot(centers, gaussian_plot, 'b-', alpha=0.8,\
                     label='Gaussian with sigma = {:.2f}'.format(RN))
            plt.plot([mean.value, mean.value], [1, 2*max(hist)],
                     label='Mean Pixel Value')
            plt.xlim(np.floor(mean.value-7.*RN.value),
                     np.ceil(mean.value+7.*RN.value))
            plt.ylim(1, 2*max(hist))
            ax.set_xlabel('Counts (ADU)', fontsize=10)
            ax.set_ylabel('Number of Pixels', fontsize=10)
            ax.grid()
            ax.legend(loc='best', fontsize=10)

    if plots:
        plotfilename = 'BiasHistogram.png'
        logger.info('Saving: {}'.format(plotfilename))
        plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
        plt.close()

    return read_noise, master_biases



##-------------------------------------------------------------------------
## Determine Dark Current
##-------------------------------------------------------------------------
def dark_current(dark_files, master_biases, plots=False, logger=None, chips=[1,2,3]):
    ndarks = len(dark_files)
    hpthresh = 0.2   # hot pixel defined as dark current of 0.2 ADU/s
    clipping_sigma = 5
    clipping_iters = 1
    plt.figure(figsize=(9,15), dpi=72)
    binsize = 1.0

    dark_table = Table(names=('filename', 'exptime', 'chip', 'mean', 'median', 'stddev', 'nhotpix'),\
                       dtype=('a64', 'f4', 'i4', 'f4', 'f4', 'f4', 'i4'))
    logger.info('Analyzing bias subtracted dark frames to measure dark current.')
    logger.info('  Determining image statistics of each dark using sigma clipping.')
    logger.info('    sigma={:d}, iters={:d}'.format(clipping_sigma, clipping_iters))
    logger.info('  Hot pixels defines as pixels with dark current > {:.2f} ADU/s'.format(hpthresh))

    for dark_file in dark_files:
        hdr = fits.getheader(dark_file, 0)
        exptime = float(hdr['DARKTIME'])
        for chip in chips:
            dark = ccdproc.fits_ccddata_reader(dark_file, unit='adu', hdu=chip)
            dark_diff = ccdproc.subtract_bias(dark, master_biases[chip])
            mean, median, stddev = stats.sigma_clipped_stats(dark_diff.data,
                                         sigma=clipping_sigma,
                                         iters=clipping_iters) * u.adu
            thresh = hpthresh*exptime # hot pixel defined as dark current of 0.2 ADU/s
            nhotpix = len(dark_diff.data.ravel()[dark_diff.data.ravel() > thresh])
            dark_table.add_row([dark_file, exptime, chip, mean, median, stddev, nhotpix])

    logger.debug('  Fitting line to levels as a function of exposure time')
    line = models.Linear1D(intercept=0, slope=0)
    line.intercept.fixed = True
    fitter = fitting.LinearLSQFitter()
    dc_fit = {}

    longest_exptime = int(max(dark_table['exptime']))
    long_dark_table = dark_table[np.array(dark_table['exptime'], dtype=int) == longest_exptime]

    dark_stats = {}
    for chip in chips:
        dc_fit[chip] = fitter(line, dark_table[dark_table['chip'] == chip]['exptime'],\
                              dark_table[dark_table['chip'] == chip]['mean'])
        dark_current = dc_fit[chip].slope.value * u.adu/u.second
        thischip = long_dark_table[long_dark_table['chip'] == chip]
        nhotpix = int(np.mean(thischip['nhotpix']))
        nhotpixstd = int(np.std(thischip['nhotpix']))
        logger.info('For chip #{:d}:'.format(chip))
        logger.info('  Dark Current = {:.2f} ADU/600s'.format(dark_current.value*600.))
        logger.info('  Found {:d} +/- {:d} hot pixels'.format(nhotpix, nhotpixstd))
        dark_stats[chip] = [dark_current, nhotpix, nhotpixstd]

    ##-------------------------------------------------------------------------
    ## Plot Dark Frame Levels
    ##-------------------------------------------------------------------------
    if plots:
        logger.info('Generating figure with dark current fits')
        plt.figure(figsize=(15,9), dpi=72)
        ax = plt.subplot(111)
        ax.plot(dark_table[dark_table['chip'] == 1]['exptime'],\
                dark_table[dark_table['chip'] == 1]['mean'],\
                'bo',\
                label='mean count level in ADU (B)',\
                alpha=1.0)
        ax.plot([0, 1000],\
                [dc_fit[1](0), dc_fit[1](1000)],\
                'b-',\
                label='dark current (B) = {:.2f} ADU/600s'.format(dc_fit[1].slope.value*600.),\
                alpha=0.3)
        ax.plot(dark_table[dark_table['chip'] == 2]['exptime'],\
                dark_table[dark_table['chip'] == 2]['mean'],\
                'go',\
                label='mean count level in ADU (G)',\
                alpha=1.0)
        ax.plot([0, 1000],\
                [dc_fit[2](0), dc_fit[2](1000)],\
                'g-',\
                label='dark current (G) = {:.2f} ADU/600s'.format(dc_fit[1].slope.value*600.),\
                alpha=0.3)
        ax.plot(dark_table[dark_table['chip'] == 3]['exptime'],\
                dark_table[dark_table['chip'] == 3]['mean'],\
                'ro',\
                label='mean count level in ADU (R)',\
                alpha=1.0)
        ax.plot([0, 1000],\
                [dc_fit[3](0), dc_fit[3](1000)],\
                'r-',\
                label='dark current (R) = {:.2f} ADU/600s'.format(dc_fit[3].slope.value*600.),\
                alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        plt.xlim(-0.02*max(dark_table['exptime']), 1.10*max(dark_table['exptime']))
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
        plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
        plt.close()
        logger.info('  Done.')

    return dark_stats



##-------------------------------------------------------------------------
## Determine Gain
##-------------------------------------------------------------------------
def gain(flat_files, master_biases, plots=False, logger=None, chips=[1,2,3]):
    clipping_sigma = 5
    clipping_iters = 1
    flat_table = Table(names=('filename', 'exptime', 'chip', 'mean', 'median', 'stddev', 'gain'),\
                       dtype=('a64', 'f4', 'i4', 'f4', 'f4', 'f4', 'f4'))

    flats = {1: [], 2: [], 3: []}
    logger.info('Reading in all flat files')
    for flat_file in flat_files:
        logger.debug('  Reading {}'.format(flat_file))
        hdr = fits.getheader(flat_file, 0)
        exptime = float(hdr['EXPTIME'])
        for chip in chips:
            flat = ccdproc.fits_ccddata_reader(flat_file, unit='adu', hdu=chip)
            flat_bs = ccdproc.subtract_bias(flat, master_biases[chip])
            flat_1s = flat_bs.divide(exptime * u.second)
            flats[chip].append(flat_1s)

    master_flats = {1: None, 2: None, 3: None}
    for chip in chips:
        logger.info('Making master flat for chip {:d}'.format(chip))
        combined = ccdproc.combine(flats[chip], method='average',
                                   sigma_clip=True,
                                   sigma_clip_low_thresh=clipping_sigma,
                                   sigma_clip_high_thresh=clipping_sigma,
                                  )
        norm_factor = np.percentile(combined.data, 99.8) * u.adu / u.second
        master_flats[chip] = combined.divide(norm_factor)
        
        logger.info('  Masking low SNR pixels')
        dev = ccdproc.background_deviation_filter(master_flats[chip].data, bbox=11)
        snr = master_flats[chip].data/dev
        mask = (snr <= 10)
        nmasked = np.sum(np.array(mask.ravel(), dtype=int))
        master_flats[chip].mask = mask

        logger.debug('  Writing master flat to file')
        master_flats[chip].write('master_flat{:d}.fits'.format(chip), overwrite=True)

        mean, median, stddev = stats.sigma_clipped_stats(master_flats[chip].data,
                                     mask = mask,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * master_flats[chip].unit
        logger.debug('  Masked {:d} pixels in master flat'.format(nmasked))
        logger.debug('  MasterFlat[{:d}] (mean, med, std) = '\
                     '{:.1f}, {:.1f}, {:.2f} {}'.format(\
                     chip, mean.value, median.value, stddev.value, master_flats[chip].unit))

        
        for flat in flats[chip]:
            flat.mask = mask
            flat_flattened = flat.divide(master_flats[chip])
            mean, median, stddev = stats.sigma_clipped_stats(flat_flattened.data,
                                         mask = mask,
                                         sigma=clipping_sigma,
                                         iters=clipping_iters) * flat_flattened.unit
            gain = median.value / stddev.value**2
            logger.debug('  Flat[{:d}] (mean, med, std, gain) = '\
                         '{:.1f}, {:.1f}, {:.2f} {}, {:.2f}'.format(\
                         chip, mean.value, median.value, stddev.value, flat_flattened.unit, gain))

            flat_table.add_row([flat_file, exptime, chip, mean, median, stddev, gain])

#             flat_flattened.write('flat_flattened{:d}.fits'.format(chip), overwrite=True)
#             sys.exit(0)

    print(flat_table)


if __name__ == '__main__':
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
    parser.add_argument("--noplots",
        action="store_true", dest="noplots",
        default=False, help="Do not make plots.")
    ## add arguments
    parser.add_argument(
        type=str, dest="input",
        help="A text file which contains the list of files to use in the analysis.")
    args = parser.parse_args()

    plots = not args.noplots

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


    lists = get_file_list(args.input)
    RN, master_biases = read_noise(lists['bias'], plots=plots, logger=logger, chips=[1])
    print(RN)
    DC = dark_current(lists['dark'], master_biases, plots=plots, logger=logger, chips=[1])
    print(DC)
    gain(lists['flat'], master_biases, plots=plots, logger=logger, chips=[1])
