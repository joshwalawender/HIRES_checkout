#!/usr/env/python

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column
from astropy import stats
from ccdproc import CCDData, Combiner, subtract_bias

## Suppress astropy log
from astropy import log
log.setLevel('ERROR')
# log.disable_warnings_logging()

import ktl

class HIRESError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


##-------------------------------------------------------------------------
## HIRES Instrument Object
##-------------------------------------------------------------------------
class HIRES(object):
    def __init__(self, logger=None):
        self.logger = logger
        self.hires = ktl.Service('hires')
        self.hiccd = ktl.Service('hiccd')
        ## Set OBSERVER to "HIRES Engineering"
        self.hiccd.write('OBSERVER', 'HIRES Engineering')
        ## Set frame numebr to 1
        self.hiccd.write('FRAMENO', 1)

    def goi(self, n=1):
        notbusy = ktl.waitFor('($hiccd.OBSERVIP == false)', timeout=30)
        if notbusy:
            for i in range(0,n):
                self.info('Exposing ({:d} of {:d}) ...'.format(i, n))
                exptime = float(self.hiccd.read('TTIME'))
                self.info('  Exposure Time = {:.1f} s'.format(exptime))
                self.info('  Object = {}'.format(
                          self.hiccd.read('OBJECT')))
                self.info('  Type = {}'.format(
                          self.hiccd.read('OBSTYPE')))
                self.hiccd.write('EXPOSE', True)
                self.info('  Waiting for exposure to finish ...')
                done = ktl.waitFor('($hiccd.OBSERVIP == false)', timeout=30+exptime)
                if done:
                    self.info('  Done.')
                else:
                    self.error('Timed out waiting for exposure to finish')
                    raise HIRESError('Timed out waiting for exposure to finish')

    def take_bias(self, nbiases=1):
        self.hiccd.write('OBJECT', 'Bias')
        self.hiccd.write('OBSTYPE', 'Bias')
        self.hiccd.write('AUTOSHUT', False)
        self.hiccd.write('TTIME', 0)
        self.goi(n=nbiases)

    ##-------------------------------------------------------------------------
    ## Logging Convenience Methods
    ##-------------------------------------------------------------------------
    def debug(self, msg):
        if self.logger:
            self.logger.debug(msg)
        else:
            print('  DEBUG: {}'.format(msg))

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print('   INFO: {}'.format(msg))

    def warning(self, msg):
        if self.logger:
            self.logger.warning(msg)
        else:
            print('WARNING: {}'.format(msg))

    def error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print('  ERROR: {}'.format(msg))


##-------------------------------------------------------------------------
## HIRES Image Object
##-------------------------------------------------------------------------
class HIRESimage(object):
    def __init__(self, fits_file=None, logger=None):
        self.logger = logger
        self.b = None
        self.g = None
        self.r = None
        self.header = None
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
        debug('Reading {}'.format(fits_file))
        self.b = CCDData.read(fits_file, unit=u.adu, hdu=1)
        self.g = CCDData.read(fits_file, unit=u.adu, hdu=2)
        self.r = CCDData.read(fits_file, unit=u.adu, hdu=3)
        self.header = fits.getheader(fits_file, 0)

    def load_from_CCDData(self, b, g, r):
        debug('Loading data from CCDData objects')
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
    ## Logging Convenience Methods
    ##-------------------------------------------------------------------------
    def debug(msg):
        if self.logger:
            self.logger.debug(msg)
        else:
            print('  DEBUG: {}'.format(msg))

    def info(msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print('   INFO: {}'.format(msg))

    def warning(msg):
        if self.logger:
            self.logger.warning(msg)
        else:
            print('WARNING: {}'.format(msg))

    def error(msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print('  ERROR: {}'.format(msg))


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
        info('Taking {} bias frames'.format(nbiases))
        if fake:
            ## NOTE: this assumes nbiases=10 at the moment
            nbiases = 10
            info('  Fake keyword invoked.  No data will be taken.')
            info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            bias_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(1,nbiases+1,1)]
            for bias_file in bias_files:
                debug('  Bias: {}'.format(bias_file))
        else:
            ## Take bias frames here
            bias_files = []

        info('  Obtained {} bias files'.format(len(bias_files)))
        return bias_files

    def take_darks(self, ndarks=5, exptimes=[1, 60, 600], fake=False):
        '''Script to use KTL and keywords to take n dark frames at the
        specified exposure times and record the resulting filenames to a list.
        '''
        info('Taking dark frames')
        if fake:
            ## NOTE: this assumes ndarks=5 and exptimes=[1, 60, 600]
            ndarks=5
            exptimes=[1, 60, 600]
            info('  Fake keyword invoked.  No data will be taken.')
            info('  Returning generic file names')
            base_path = os.path.join('/', 'Volumes', 'Internal_1TB', 'HIRES_Data')
            start_index = 11
            dark_files = [os.path.join(base_path, 'hires{:04d}.fits'.format(i))\
                          for i in range(start_index,start_index+ndarks*(len(exptimes)),1)]
            for dark_file in dark_files:
                debug('  {}'.format(dark_file))
            return dark_files
        dark_files = []
        ## Take bias frames here
        return dark_files
