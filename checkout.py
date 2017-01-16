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
    logger.info('Analyzing noise in bias frames to determine read noise')
    nbiases = len(bias_files)
    clipping_sigma = 5
    clipping_iters = 1
    if plots:
        plt.figure(figsize=(11,len(chips)*5), dpi=72)
        binsize = 1.0
    master_biases = {}
    read_noise = {}
    for chip in chips:
        logger.info('  Analyzing Chip {:d}'.format(chip))
        if plots:
            ax = plt.subplot(len(chips),1,chip)
            color = {1: 'B', 2: 'G', 3: 'R'}
        biases = []
        for bias_file in bias_files:
            logger.debug('  Reading bias: {}[{}]'.format(bias_file, chip))
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
        master_bias = ccdproc.combine(biases, combine='average',
                                      sigma_clip=True,
                                      sigma_clip_low_thresh=clipping_sigma,
                                      sigma_clip_high_thresh=clipping_sigma)
        master_biases[chip] = master_bias
        mean, median, stddev = stats.sigma_clipped_stats(master_bias.data,
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        mode = get_mode(master_bias)
        logger.info('  Master Bias[{:d}] (mean, med, mode, std) = '\
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
        logger.info('  Read Noise[{:d}] is {:.2f}'.format(chip, RN))

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
                    align='center', width=0.7*binsize, log=True, color='{}'.format(color[chip].lower()),
                    alpha=0.5,
                    label='{} CCD Pixel Count Histogram'.format(color[chip]))
            plt.plot(centers, gaussian_plot, '{}-'.format(color[chip].lower()), alpha=0.8,\
                     label='Gaussian with sigma = {:.2f}'.format(RN))
            plt.plot([mean.value, mean.value], [1, 2*max(hist)], 'k-',
                     label='Mean Pixel Value')
            plt.xlim(np.floor(mean.value-15.*RN.value),
                     np.ceil(mean.value+15.*RN.value))
            plt.ylim(1, 2*max(hist))
            ax.set_xlabel('Counts (ADU)', fontsize=10)
            ax.set_ylabel('Number of Pixels', fontsize=10)
            ax.grid()
            ax.legend(loc='upper left', fontsize=10)

    if plots:
        plotfilename = 'BiasHistogram.png'
        logger.info('  Saving: {}'.format(plotfilename))
        plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
        plt.close()

    return read_noise, master_biases



##-------------------------------------------------------------------------
## Determine Dark Current
##-------------------------------------------------------------------------
def dark_current(dark_files, master_biases, plots=False, logger=None, chips=[1,2,3]):
    ndarks = len(dark_files)
    hpthresh = 0.5  # hot pixel defined as dark current of 0.5 ADU/s ~ 1 e-/s
    clipping_sigma = 5
    clipping_iters = 1
    binsize = 1.0

    dark_table = Table(names=('filename', 'exptime', 'chip', 'mean', 'median', 'stddev', 'nhotpix'),\
                       dtype=('a64', 'f4', 'i4', 'f4', 'f4', 'f4', 'i4'))
    logger.info('Analyzing bias subtracted dark frames to measure dark current.')
    logger.debug('  Determining image statistics of each dark using sigma clipping.')
    logger.debug('    sigma={:d}, iters={:d}'.format(clipping_sigma, clipping_iters))
    logger.info('  Hot pixels are defined as pixels with dark current > {:.2f} ADU/s'.format(hpthresh))

    for dark_file in dark_files:
        hdr = fits.getheader(dark_file, 0)
        exptime = float(hdr['DARKTIME'])
        for chip in chips:
            logger.debug('  Reading dark: {}[{}]'.format(dark_file, chip))
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
#         print(fitter.fit_info)
        dark_current = dc_fit[chip].slope.value * u.adu/u.second
        thischip = long_dark_table[long_dark_table['chip'] == chip]
        nhotpix = int(np.mean(thischip['nhotpix'])) * u.pix
        nhotpixstd = int(np.std(thischip['nhotpix'])) / np.sqrt(len(thischip['nhotpix'])) * u.pix
        logger.info('  Analyzing Chip {:d}'.format(chip))
        logger.info('  Dark Current[{:d}] = {:.4f} ADU/600s'.format(chip, dark_current.value*600.))
        logger.info('  N Hot Pixels[{:d}] = {:.0f} +/- {:.0f}'.format(chip, nhotpix, nhotpixstd))
        dark_stats[chip] = [dark_current, nhotpix, nhotpixstd]

    ##-------------------------------------------------------------------------
    ## Plot Dark Frame Levels
    ##-------------------------------------------------------------------------
    if plots:
        color = {1: 'B', 2: 'G', 3: 'R'}
        plt.figure(figsize=(11,len(chips)*5), dpi=72)
        for chip in chips:
            ax = plt.subplot(len(chips),1,chip)
            ax.plot(dark_table[dark_table['chip'] == chip]['exptime'],\
                    dark_table[dark_table['chip'] == chip]['mean'],\
                    '{}o'.format(color[chip].lower()),\
                    label='mean count level in ADU ({})'.format(color[chip]),\
                    alpha=1.0)
            ax.plot([0, 1000],\
                    [dc_fit[chip](0), dc_fit[chip](1000)],\
                    '{}-'.format(color[chip].lower()),\
                    label='dark current ({}) = {:.2f} ADU/600s'.format(\
                          color[chip], dc_fit[chip].slope.value*600.),\
                    alpha=0.3)
            plt.xlim(-0.02*max(dark_table['exptime']), 1.10*max(dark_table['exptime']))
            min_level = np.floor(min(dark_table['mean']))
            max_level = np.ceil(max(dark_table['mean']))
            plt.ylim(min([0,min_level]), max_level)
            ax.set_xlabel('Exposure Time (s)')
            ax.set_ylabel('Dark Level (ADU)')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid()

        plotfilename = 'DarkCurrent.png'
        logger.info('  Saving: {}'.format(plotfilename))
        plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
        plt.close()

    return dark_stats



##-------------------------------------------------------------------------
## Determine Gain
##-------------------------------------------------------------------------
def gain(flat_files, master_biases, read_noise=None, plots=False, logger=None, chips=[1,2,3]):
    clipping_sigma = 5
    clipping_iters = 1
    flat_table = Table(names=('filename', 'EXPTIME', 'TTIME'),\
                       dtype=('a64', 'f4', 'f4'))

    logger.info('Fitting model to signal vs. variance data to derive gain')
    logger.debug('  Reading headers for all flat files')
    for flat_file in flat_files:
        logger.debug('  Reading {}'.format(flat_file))
        hdr = fits.getheader(flat_file, 0)
        exptime = float(hdr['EXPTIME'])
        ttime = float(hdr['TTIME'])
        flat_table.add_row([flat_file, exptime, ttime])

    bytime = flat_table.group_by('TTIME')
    ttimes = sorted(set(flat_table['TTIME']))
    signal = {1: [], 2: [], 3: []}
    variance = {1: [], 2: [], 3: []}
    for ttime in ttimes:
        exps = bytime.groups[bytime.groups.keys['TTIME'] == ttime]
        nexps = len(exps)
        logger.debug('  Measuring statistics for {:.0f} s flats'.format(float(ttime)))
        for i in np.arange(0,nexps,2):
            if i+1 < nexps:
                flat_fileA = exps['filename'][i].decode('utf8')
                flat_fileB = exps['filename'][i+1].decode('utf8')
                for chip in chips:
                    logger.debug('  Reading flat: {}[{}]'.format(flat_fileA, chip))
                    expA = ccdproc.fits_ccddata_reader(flat_fileA, unit='adu', hdu=chip)
                    expA_bs = ccdproc.subtract_bias(expA, master_biases[chip])
                    meanA, medA, stdA = stats.sigma_clipped_stats(expA_bs.data,
                                              sigma=clipping_sigma,
                                              iters=clipping_iters)
                    logger.debug('  Reading flat: {}[{}]'.format(flat_fileB, chip))
                    expB = ccdproc.fits_ccddata_reader(flat_fileB, unit='adu', hdu=chip)
                    expB_bs = ccdproc.subtract_bias(expB, master_biases[chip])
                    meanB, medB, stdB = stats.sigma_clipped_stats(expB_bs.data,
                                              sigma=clipping_sigma,
                                              iters=clipping_iters)
                    logger.debug('  Forming A-B difference pair for variance '\
                                 'measurement with {:.1f} s exposure'.format(
                                 float(ttime)))
                    ratio = meanA/meanB
                    expB_scaled = expB_bs.multiply(ratio)
                    diff = expA_bs.subtract(expB_scaled)
                    mean, med, std = stats.sigma_clipped_stats(diff.data,
                                              sigma=clipping_sigma,
                                              iters=clipping_iters)
                    logger.debug('  Signal Level = {:.1f}'.format(meanA))
                    logger.debug('  Variance = {:.1f}'.format(std**2/2.0))
                    variance[chip].append(std**2/2.0)
                    signal[chip].append(meanA)

    ## Fit model to variance vs. signal
    ## var = RN^2 + 1/g S + k^2 S^2
    gainfits = {}
    g = {}
    gerr = {}
    for chip in chips:
        if read_noise:
            poly = models.Polynomial1D(degree=2,\
                                       c0=read_noise[chip].to(u.adu).value)
            poly.c0.fixed = True
        else:
            poly = models.Polynomial1D(degree=2)
        poly.c2.min = 0.0
        fitter = fitting.LevMarLSQFitter()
        gainfits[chip] = fitter(poly, signal[chip], variance[chip])
        perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
        ksq = gainfits[chip].c2.value
        ksqerr = perr[1]
        logger.info('  k^2[{}] = {:.2e} +/- {:.2e} e/ADU'.format(chip, ksq, ksqerr))
        g[chip] = gainfits[chip].c1**-1 * u.electron/u.adu
        gerr[chip] = gainfits[chip].c1**-2 * perr[0] * u.electron/u.adu
        logger.info('  Gain[{}] = {:.2f} +/- {:.2f} e/ADU'.format(chip,
                    g[chip].value, gerr[chip].value))


    ##-------------------------------------------------------------------------
    ## Plot Flat Statistics
    ##-------------------------------------------------------------------------
    if plots:
        logger.info('  Generating figure with flat statistics and gain fits')
        plt.figure(figsize=(11,len(chips)*5), dpi=72)
        color = {1: 'B', 2: 'G', 3: 'R'}
        for chip in chips:
            ax = plt.subplot(len(chips),1,chip)
            ax.plot(signal[chip],\
                    variance[chip],\
                    '{}o'.format(color[chip].lower()),\
                    alpha=1.0)
            sig_fit = np.linspace(min(signal[chip]), max(signal[chip]), 50)
            var_fit = [gainfits[chip](x) for x in sig_fit]
            ax.plot(sig_fit, var_fit,\
                    '{}-'.format(color[chip].lower()),\
                    label='Gain={:.2f} +/- {:.2f} e/ADU'.format(
                          g[chip].value, gerr[chip].value),
                    alpha=0.7)
            ax.set_ylabel('Variance')
            ax.set_xlabel('Mean Level (ADU)')
            ax.grid()
            ax.legend(loc='upper left', fontsize=10)

        plotfilename = 'FlatStats.png'
        logger.info('  Saving: {}'.format(plotfilename))
        plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
        plt.close()
        logger.info('  Done.')

    return g, gerr


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

    chips = [1, 2, 3]
    lists = get_file_list(args.input)
    RNC, master_biases = read_noise(lists['bias'],
                                   plots=plots, logger=logger, chips=chips)
    DCC = dark_current(lists['dark'], master_biases,
                      plots=plots, logger=logger, chips=chips)
    g, gerr = gain(lists['flat'], master_biases, read_noise=RNC,
                   plots=plots, logger=logger, chips=chips)

    for chip in chips:
        RNe = RNC[chip] * g[chip]
        DCe = DCC[chip][0] * g[chip]
        print('Chip {:d}'.format(chip))
        print('  Read Noise[{:d}]   = {:.1f}'.format(chip, RNe))
        print('  Dark Current[{:d}] = {:.4f}'.format(chip, DCe))
        print('  N Hot Pixels[{:d}] = {:.0f} +/- {:.0f}'.format(chip, DCC[chip][1], DCC[chip][2]))
        print('  Gain[{:d}]         = {:.2f} +/- {:.2f} e/ADU'.format(chip, g[chip].value, gerr[chip].value))
