#!/usr/env/python

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging

from datetime import datetime as time
from datetime import timedelta as dt

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
