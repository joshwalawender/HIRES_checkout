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
    dates = []
    for file in files:
        hdr = fits.getheader(file, 0)
        date = hdr.get('DATE-OBS')
        if not date in dates:
            dates.append(date)
        if hdr.get('OBSTYPE').strip() == 'Bias':
            bias_files.append(file)
        elif hdr.get('OBSTYPE').strip() == 'Dark':
            dark_files.append(file)
        elif hdr.get('OBSTYPE').strip() == 'IntFlat':
            flat_files.append(file)

    dict = {'bias': bias_files, 'dark': dark_files, 'flat': flat_files,
            'dates': dates}

    return dict


##-------------------------------------------------------------------------
## Determine Read Noise
##-------------------------------------------------------------------------
def read_noise(lists, plots=False, logger=None, chips=[1,2,3],
               clipping_sigma=5, clipping_iters=3):
    '''
    '''
    buf = 200  # pixel buffer for stats
    bias_files = lists['bias']
    logger.info('Analyzing noise in bias frames to determine read noise')
    nbiases = len(bias_files)
    if plots:
        plt.figure(figsize=(11,len(chips)*5), dpi=72)
        binsize = 1.0
    master_biases = {}
    read_noise = {}
    for chip in chips:
        logger.info('  Analyzing Chip {:d}'.format(chip))
        if plots:
            ax = plt.subplot(len(chips),1,chip)
            color = {1: 'B', 2: 'G', 3: 'R', 4: 'Y'}
        biases = []
        for bias_file in bias_files:
            logger.debug('  Reading bias: {}[{}]'.format(bias_file, chip))
            if bias_file == bias_files[0]:
                bias0 = ccdproc.fits_ccddata_reader(bias_file, unit='adu', hdu=chip)
                ny, nx = bias0.data.shape
                mean, median, stddev = stats.sigma_clipped_stats(
                                             bias0.data[buf:ny-buf,buf:nx-buf],
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
        ny, nx = master_bias.data.shape
        mean, median, stddev = stats.sigma_clipped_stats(
                                     master_bias.data[buf:ny-buf,buf:nx-buf],
                                     sigma=clipping_sigma,
                                     iters=clipping_iters) * u.adu
        mode = get_mode(master_bias)
        logger.info('  Master Bias[{:d}] (mean, med, mode, std) = '\
                    '{:.1f}, {:.1f}, {:d}, {:.2f}'.format(\
                    chip, mean.value, median.value, mode, stddev.value))

        diff = bias0.subtract(master_bias)
        ny, nx = diff.data.shape
        mean, median, stddev = stats.sigma_clipped_stats(
                                     diff.data[buf:ny-buf,buf:nx-buf],
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
            centers = (bins[:-1] + bins[1:]) / 2 * u.adu
            gaussian = models.Gaussian1D(amplitude=max(hist),\
                                         mean=mean,\
                                         stddev=RN)
            gaussian_plot = [gaussian(x) for x in centers]
            plt.bar(centers.value, hist,
                    align='center', width=0.7*binsize, log=True,
                    color='{}'.format(color[chip].lower()), alpha=0.5,
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
def dark_current(lists, master_biases, plots=False, logger=None,
                 chips=[1,2,3], clipping_sigma=5, clipping_iters=3, hpthresh=1.0):
    dark_files = lists['dark']
    ndarks = len(dark_files)
    binsize = 1.0
    buf = 200  # pixel buffer for stats

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
            ny, nx = dark_diff.data.shape
            mean, median, stddev = stats.sigma_clipped_stats(
                                         dark_diff.data[buf:ny-buf,buf:nx-buf],
                                         sigma=clipping_sigma,
                                         iters=clipping_iters) * u.adu
            thresh = hpthresh*exptime
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
        dc_fit[chip] = fitter(line, 
                              dark_table[dark_table['chip'] == chip]['exptime'],
                              dark_table[dark_table['chip'] == chip]['mean'])
        dark_current = dc_fit[chip].slope.value * 3600 * u.adu/u.hour
        thischip = long_dark_table[long_dark_table['chip'] == chip]
        nhotpix = int(np.mean(thischip['nhotpix'])) * u.pix
        nhotpixstd = int(np.std(thischip['nhotpix'])) / np.sqrt(len(thischip['nhotpix'])) * u.pix
        logger.info('  Analyzing Chip {:d}'.format(chip))
        logger.info('  Dark Current[{:d}] = {:.1f} ADU/hr'.format(chip, dark_current.value))
        logger.info('  N Hot Pixels[{:d}] = {:.0f} +/- {:.0f}'.format(chip, nhotpix, nhotpixstd))
        dark_stats[chip] = [dark_current, nhotpix, nhotpixstd]

    ##-------------------------------------------------------------------------
    ## Plot Dark Frame Levels
    ##-------------------------------------------------------------------------
    if plots:
        color = {1: 'B', 2: 'G', 3: 'R', 4: 'Y'}
        plt.figure(figsize=(11,len(chips)*5), dpi=72)
        for chip in chips:
            ax = plt.subplot(len(chips),1,chip)
            ax.plot(dark_table[dark_table['chip'] == chip]['exptime'],\
                    dark_table[dark_table['chip'] == chip]['mean'],\
                    '{}o'.format(color[chip].lower()),\
                    label='mean count level in ADU ({})'.format(color[chip]),\
                    alpha=1.0)
            ax.plot([0, longest_exptime],\
                    [dc_fit[chip](0), dc_fit[chip](longest_exptime)],\
                    '{}-'.format(color[chip].lower()),\
                    label='dark current ({}) = {:.2f} ADU/hr'.format(\
                          color[chip], dark_stats[chip][0].value),\
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
def gain(lists, master_biases, read_noise=None, plots=False, logger=None,
         chips=[1,2,3], clipping_sigma=5, clipping_iters=3, aduthreshold=25000):
    buf = 200  # pixel buffer for stats
    flat_files = lists['flat']
    flat_table = Table()
    logger.info('Fitting model to signal vs. variance data to derive gain')
    logger.info('  Reading flat files')
    exptimes = []
    ttimes = []
    ccds = {}
    signal = {}
    for ffi,flat_file in enumerate(flat_files):
        logger.debug('  Reading {:d}/{:d}: {}'.format(ffi, len(flat_files), flat_file))
        hdr = fits.getheader(flat_file, 0)
        exptimes.append(float(hdr['EXPTIME']))
        ttimes.append(float(hdr['TTIME']))
        ccds[flat_file] = {}
        for chip in chips:
            if not chip in signal.keys():
                signal[chip] = []
            exp = ccdproc.fits_ccddata_reader(flat_file, unit='adu', hdu=chip)
            ccds[flat_file][chip] = ccdproc.subtract_bias(exp, master_biases[chip])
            ny, nx = ccds[flat_file][chip].data.shape
            mean, med, std = stats.sigma_clipped_stats(
                                   ccds[flat_file][chip].data[buf:ny-buf,buf:nx-buf],
                                   sigma=clipping_sigma,
                                   iters=clipping_iters)
            signal[chip].append(mean)
    flat_table.add_columns([Column(flat_files, 'filename', dtype='a128'),
                            Column(exptimes, 'EXPTIME', dtype='f4'),
                            Column(ttimes, 'TTIME', dtype='f4')])
    for chip in chips:
        flat_table.add_column(Column(signal[chip], 'signal{:d}'.format(chip), dtype='f4'))

    bytime = flat_table.group_by('TTIME')
    ttimes = sorted(set(flat_table['TTIME']))
    signal = {}
    variance = {}
    signal_times = {}
    for ttime in ttimes:
        exps = bytime.groups[bytime.groups.keys['TTIME'] == ttime]
        nexps = len(exps)
        logger.info('  Measuring statistics for {:.0f} s flats'.format(float(ttime)))
        for i in np.arange(0,nexps,2):
            if i+1 < nexps:
                try:
                    flat_fileA = exps['filename'][i].decode('utf8')
                    flat_fileB = exps['filename'][i+1].decode('utf8')
                except:
                    flat_fileA = exps['filename'][i]
                    flat_fileB = exps['filename'][i+1]
                for chip in chips:
                    if not chip in signal.keys():
                        signal[chip] = []
                        variance[chip] = []
                        signal_times[chip] = []
                    expA = ccds[flat_fileA][chip]
                    expB = ccds[flat_fileB][chip]
                    meanA = exps[i]['signal{:d}'.format(chip)]
                    meanB = exps[i+1]['signal{:d}'.format(chip)]
                    ratio = meanA/meanB
                    logger.debug('  Forming A-B difference pair with a scaling ratio of {:.3f}'.format(
                                 float(ttime), ratio))
                    expB_scaled = expB.multiply(ratio)
                    diff = expA.subtract(expB_scaled)
#                     if i==0:
#                         diff.write('diff_{:d}s.fits'.format(int(ttime)))
                    ny, nx = ccds[flat_file][chip].data.shape
                    mean, med, std = stats.sigma_clipped_stats(
                                        diff.data[buf:ny-buf,buf:nx-buf],
                                        sigma=clipping_sigma,
                                        iters=clipping_iters)
                    logger.debug('  Signal Level = {:.2f}'.format(meanA))
                    logger.debug('  Variance = {:.2f}'.format(std**2/2.0))
                    variance[chip].append(std**2/2.0)
                    signal[chip].append(meanA)
                    signal_times[chip].append(exps[i]['EXPTIME'])

    ## Fit model to variance vs. signal
    ## var = RN^2 + 1/g S + k^2 S^2
    gainfits = {}
    g = {}
    gerr = {}
    linearity_fit = {}
    linearity_fitp = {}
    mask = {}
    for chip in chips:
        mask[chip] = np.array(np.array(signal[chip]) > aduthreshold)

        ## Fit Gain with Polynomial Model
        if read_noise:
            poly = models.Polynomial1D(degree=2,\
                                       c0=read_noise[chip].to(u.adu).value)
            poly.c0.fixed = True
        else:
            poly = models.Polynomial1D(degree=2)
        poly.c2.min = 0.0
        fitter = fitting.LevMarLSQFitter()
        y = np.array(variance[chip])[~mask[chip]]
        x = np.array(signal[chip])[~mask[chip]]
        logger.info(f'{chip} Variance')
        logger.info(y)
        logger.info(f'{chip} Signal')
        logger.info(x)
        gainfits[chip] = fitter(poly, x, y)
        logger.info(gainfits[chip])
        perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
        ksq = gainfits[chip].c2.value
        ksqerr = perr[1]
        logger.info('  k^2[{}] = {:.2e} +/- {:.2e} e/ADU'.format(chip, ksq, ksqerr))
        g[chip] = gainfits[chip].c1**-1 * u.electron/u.adu
        gerr[chip] = gainfits[chip].c1**-2 * perr[0] * u.electron/u.adu
        logger.info('  Gain[{}] = {:.2f} +/- {:.2f} e/ADU'.format(chip,
                    g[chip].value, gerr[chip].value))

        ## Fit Gain with Linear Model
        ##  var_ADU = (RN/g)^2 + 1/g * C_ADU
        ##  var_ADU = (RN_ADU)^2 + 1/g * C_ADU
#         line = models.Linear1D(intercept=read_noise[chip].to(u.adu).value**2,
#                                slope=0.5)
#         line.intercept.fixed = True
#         fitter = fitting.LinearLSQFitter()
#         y = np.array(variance[chip])[~mask[chip]]
#         x = np.array(signal[chip])[~mask[chip]]
#         gainfits[chip] = fitter(line, x, y)
#         g[chip] = 1./gainfits[chip].slope.value * u.electron/u.adu
#         logger.info('  Gain[{}] = {:.2f}'.format(chip, g[chip].value))

        ## Fit Linearity
        line = models.Linear1D(intercept=0, slope=500)
        line.intercept.fixed = True
        fitter = fitting.LinearLSQFitter()
        x = np.array(signal_times[chip])[~mask[chip]]
        y = np.array(signal[chip])[~mask[chip]]
        logger.info(f'{chip} Signal Times')
        logger.info(x)
        logger.info(f'{chip} Signal')
        logger.info(y)
        linearity_fit[chip] = fitter(line, x, y)
        logger.info(linearity_fit[chip])

        ## Fit Linearity with Polynomial
        poly = models.Polynomial1D(degree=2)
        fitter = fitting.LevMarLSQFitter()
        linearity_fitp[chip] = fitter(poly, x, y)
        logger.info(linearity_fitp[chip])

    ##-------------------------------------------------------------------------
    ## Plot Flat Statistics
    ##-------------------------------------------------------------------------
    if plots:
        for type in ['', '_log']:
            logger.info('  Generating figure with flat statistics and gain fits')
            plt.figure(figsize=(11,len(chips)*5), dpi=72)
            color = {1: 'B', 2: 'G', 3: 'R', 4: 'Y'}
            for chip in chips:
                ax = plt.subplot(len(chips),1,chip)

                x = np.array(signal[chip])[mask[chip]]
                y = np.array(variance[chip])[mask[chip]]
                ax.plot(x, y, '{}o'.format(color[chip].lower()), alpha=0.3,\
                        markersize=5, markeredgewidth=0)

                x = np.array(signal[chip])[~mask[chip]]
                y = np.array(variance[chip])[~mask[chip]]
                if type == '_log':
                    ax.semilogx(x, y, '{}o'.format(color[chip].lower()), alpha=1.0,\
                            markersize=8, markeredgewidth=0)

                    sig_fit = np.linspace(min(signal[chip]), max(signal[chip]), 50)
                    var_fit = [gainfits[chip](x) for x in sig_fit]
                    ax.semilogx(sig_fit, var_fit,\
                            '{}-'.format(color[chip].lower()),\
                            label='Gain={:.2f} +/- {:.2f} e/ADU'.format(
                                  g[chip].value, gerr[chip].value),
                            alpha=0.7)
                else:
                    ax.plot(x, y, '{}o'.format(color[chip].lower()), alpha=1.0,\
                            markersize=8, markeredgewidth=0)

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
            plotfilename = 'FlatStats{}.png'.format(type)
            logger.info('  Saving: {}'.format(plotfilename))
            plt.savefig(plotfilename, dpi=72, bbox_inches='tight')
            plt.close()
            logger.info('  Done.')


            logger.info('  Generating figure with linearity plot')
            plt.figure(figsize=(11,len(chips)*5), dpi=72)
            color = {1: 'B', 2: 'G', 3: 'R', 4: 'Y'}
            decrements = []
            for chip in chips:
                counts = np.array(signal[chip])
                time = np.array(signal_times[chip])
                fit_counts = [linearity_fitp[chip](t) for t in time]
                decrement = (counts-fit_counts)/counts * 100.
                decrements.extend(list(decrement))
            for chip in chips:
                ax = plt.subplot(len(chips),1,chip)
                time = np.array(signal_times[chip])
                counts = np.array(signal[chip])
                fit_counts = [linearity_fitp[chip](t) for t in time]
                y = (counts-fit_counts)/counts * 100.

                if type == '_log':
                    ax.semilogx(counts, y, '{}o'.format(color[chip].lower()), alpha=0.5,\
                            markersize=5, markeredgewidth=0)
                    ax.semilogx([min(counts), max(counts)], [0, 0], 'k-')
                else:
                    ax.plot(counts, y, '{}o'.format(color[chip].lower()), alpha=0.5,\
                            markersize=5, markeredgewidth=0)
                    ax.plot([min(counts), max(counts)], [0, 0], 'k-')

                ax.set_xlabel('Counts (ADU)')
                ax.set_ylabel('Signal Decrement (%) [(counts-fit)/counts]')
                plt.ylim(np.floor(min(decrements)), np.ceil(max(decrements)))
                ax.grid()
            plotfilename = 'Linearity{}.png'.format(type)
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
    LogFileName = 'HIREScheckout.txt'
    LogFileHandler = logging.FileHandler(LogFileName)
    LogFileHandler.setLevel(logging.INFO)
    LogFileHandler.setFormatter(LogFormat)
    logger.addHandler(LogFileHandler)

    chips = [1, 2, 3, 4]
    lists = get_file_list(args.input)
    RNC, master_biases = read_noise(lists,
                                   plots=plots, logger=logger, chips=chips)
    DCC = None
    if len(lists['dark']) > 0:
        DCC = dark_current(lists, master_biases,
                          plots=plots, logger=logger, chips=chips)
    g = None
    if len(lists['flat']) > 0:
        g, gerr = gain(lists, master_biases, read_noise=RNC,
                       plots=plots, logger=logger, chips=chips)

    for chip in chips:
        logger.info('Chip {:d}'.format(chip))
        if g is not None:
            RNe = RNC[chip] * g[chip]
            logger.info('  Read Noise[{:d}]   = {:.1f}'.format(chip, RNe))
        else:
            logger.info('  Read Noise[{:d}]   = {:.1f}'.format(chip, RNC[chip]))
        if DCC is not None:
            if g is not None:
                DCe = DCC[chip][0] * g[chip]
                logger.info('  Dark Current[{:d}] = {:.2f}'.format(chip, DCe))
            else:
                logger.info('  Dark Current[{:d}] = {:.2f}'.format(chip, DCC[chip][0]))
            logger.info('  N Hot Pixels[{:d}] = {:.0f} +/- {:.0f}'.format(chip, DCC[chip][1], DCC[chip][2]))
        if g is not None:
            logger.info('  Gain[{:d}]         = {:.2f} +/- {:.2f} e/ADU'.format(chip, g[chip].value, gerr[chip].value))
