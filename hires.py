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

import ktl

class InstrumentError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


##-------------------------------------------------------------------------
## Generic Keck Instrument Object
##-------------------------------------------------------------------------
class KeckInstrument(object):
    def __init__(self, logger=None):
        self.logger = logger
        self.name = 'Generic Keck Instrument'
        self.service_list = []
        self.services = {}
        self.keywords = {}

    def get_services(self):
        for name in self.service_list:
            self.services[name] = ktl.Service(name)
            self.keywords[name] = (self.services[name]).keywords()

    def get(self, kw):
        kwfound = False
        for name in self.services.keys():
            if kw in self.services[name].keywords():
                result = (self.services[name]).read(kw)
                kwfound = True
                break
        if not kwfound:
            raise InstrumentError('{} not in {} related services'.format(
                  kw, self.name))

    def set(self, kw, val):
        kwfound = False
        for name in self.services.keys():
            if kw in self.services[name].keywords():
                (self.services[name]).write(kw, val)
                kwfound = True
                break
        if not kwfound:
            raise InstrumentError('{} not in {} related services'.format(
                  kw, self.name))


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
## HIRES Instrument Object
##-------------------------------------------------------------------------
class HIRES(KeckInstrument):
    def __init__(self, logger=None, mode='Red'):
        assert mode.lower() in ['red', 'blue', 'r', 'b']
        self.mode = {'red': 'Red', 'blue': 'Blue', 'r': 'Red', 'b': 'Blue'}[mode.lower()]
        self.logger = logger
        super(HIRES, self).__init__(logger=self.logger)
        self.service_list = ['hires', 'hiccd']
        self.name = 'HIRES'
        self.get_services()
        self.shared_keywords = set(self.keywords['hires']) & set(self.keywords['hiccd'])
        
        self.info('Instantiated HIRES in {} mode'.format(self.mode))


    def goi(self, n=1):
        busy = bool(self.get('OBSERVIP'))
        if busy:
            self.info('Waiting for instrument ...')
            busy = not ktl.waitFor('($hiccd.OBSERVIP == false)', timeout=30)
        if not busy:
            for i in range(0,n):
                ## Check output file name
                outfile_base = self.get('OUTFILE')
                outfile_seq = int(self.get('FRAMENO'))
                outfile_name = '{}{:04d}.fits'.format(outfile_base, outfile_seq)
                outfile_path = self.get('OUTDIR')
                outfile = os.path.join(outfile_path, outfile_name)
                if os.path.exists(outfile):
                    self.warning('{} already exists'.format(outfile_name))
                    self.warning('System will copy old file to {}.old'.format(outfile_name))
                ## Begin Exposure
                self.info('Exposing ({:d} of {:d}) ...'.format(i+1, n))
                exptime = float(self.get('TTIME'))
                self.info('  Exposure Time = {:.1f} s'.format(exptime))
                self.info('  Object = "{}"'.format(self.get('OBJECT')))
                self.info('  Type = "{}"'.format(self.get('OBSTYPE')))
                tick = time.now()
                self.set('EXPOSE', True)
                self.info('  Waiting for exposure to finish ...')
                done = ktl.waitFor('($hiccd.OBSERVIP == false)', timeout=60+exptime)
                tock = time.now()
                elapsed = (tock-tick).total_seconds()
                self.debug('  Elapsed Time = {:.1f}'.format(elapsed))
                if done:
                    self.info('  File written to: {}'.format(outfile))
                    self.info('  Done ({:.1f} s elapsed)'.format(elapsed))
                else:
                    self.error('Timed out waiting for exposure to finish')
                    raise InstrumentError('Timed out waiting for exposure to finish')


    def open_covers(self):
        tick = time.now()
        self.info('Opening {} covers'.format(self.mode))
        self.set('{}COCOVER'.format(mode[0].upper()), 'open')
        self.set('ECHCOVER', 'open')
        self.set('XDCOVER', 'open')
        self.set('CO1COVER', 'open')
        self.set('CO2COVER', 'open')
        self.set('CAMCOVER', 'open')
        self.set('DARKSLID', 'open')
        tock = time.now()
        elapsed = (tock-tick).total_seconds()
        self.info('  Done ({:.1f} s elapsed)'.format(elapsed))


    def close_covers(self):
        tick = time.now()
        self.info('Closing {} covers'.format(self.mode))
        self.set('{}COCOVER'.format(mode[0].upper()), 'close')
        self.set('ECHCOVER', 'close')
        self.set('XDCOVER', 'close')
        self.set('CO1COVER', 'close')
        self.set('CO2COVER', 'close')
        self.set('CAMCOVER', 'close')
        self.set('DARKSLID', 'close')
        tock = time.now()
        elapsed = (tock-tick).total_seconds()
        self.info('  Done ({:.1f} s elapsed)'.format(elapsed))


    def take_bias(self, n=1):
        self.set('OBJECT', 'Bias')
        self.set('OBSTYPE', 'Bias')
        self.set('AUTOSHUT', False)
        self.set('TTIME', 0)
        self.goi(n=n)
        self.set('AUTOSHUT', True)


    def take_dark(self, exptime, n=1):
        self.set('OBJECT', 'Dark')
        self.set('OBSTYPE', 'Dark')
        self.set('AUTOSHUT', False)
        self.set('TTIME', exptime)
        self.goi(n=n)
        self.set('AUTOSHUT', True)


    def take_flat(self, exptime, lamp='quartz1', lampfilter='ug5', slit='B2', n=1):
        self.set('OBJECT', 'Flat')
        self.set('OBSTYPE', 'Flat')
        self.set('AUTOSHUT', True)
        self.set('TTIME', exptime)
        self.open_covers()
        self.set('LAMPNAME', lamp)
        self.set('LFILNAME', lampfilter)
        self.set('FIL1NAME', 'clear')
        self.set('FIL2NAME', 'clear')
        self.set('DECKNAME', slit)
        self.goi(n=n)
        self.set('LAMPNAME', 'none')
        self.set('AUTOSHUT', True)


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
