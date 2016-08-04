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


