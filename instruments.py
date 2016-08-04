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
        self.shared_keywords = []


    def get_services(self):
        for name in self.service_list:
            self.services[name] = ktl.Service(name)
            self.keywords[name] = (self.services[name]).keywords()
        ## Find all keywords that appear in more than one service
        allshared = []
        for i,iname in enumerate(self.service_list):
            for j in range(i+1,len(self.service_list)):
                jname = self.service_list[j]
                shared = list(set(self.keywords[iname]) & set(self.keywords[jname]))
                allshared.extend(shared)
        self.shared_keywords = set(allshared)


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

        self.slits = ['B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
                      'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5']
        self.lamps = ['none', 'off', 'ThAr1', 'ThAr2', 'quartz1', 'quartz2']
        self.lamp_filters = ['dt', 'ug1', 'gg495', 'bg12', 'bg14', 'ng3',
                             'clear', 'ug5', 'etalon', 'bg13', 'bg38']
        self.filters1 = ['bg24a', 'kv408', 'kv370', 'kv389', 'clear', 'rg610',
                         'og530', 'kv418', 'kv380', 'gg475', 'wg335', 'wg360']
        self.filters2 = ['cuso4', '6563/30', 'dt', 'clear', '5026/600', '6300/30',
                         'home', '3090/62', '6199/30', '5893/30']

        self.get_services()
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


    def set_slit(self, slit):
        assert slit.upper() in self.slits
        self.set('DECKNAME', slit.upper())


    def set_lamp(self, lamp):
        assert lamp in self.lamps
        if lamp == 'off':
            lamp = 'none'
        self.set('LAMPNAME', lamp)


    def set_lamp_filter(self, filter):
        assert filter in self.lamp_filters
        self.set('LFILNAME', lampfilter)


    def set_filter1(self, filter):
        assert filter in self.filters1
        self.set('FIL1NAME', filter)


    def set_filter2(self, filter):
        assert filter in self.filters2
        self.set('FIL2NAME', filter)


    def set_exptime(self, exptime):
        if type(exptime) is not int:
            exptime = int(exptime)
            self.warning('Exposure time must be integer.  Using {:d}'.format(exptime))
        self.set('TTIME', exptime)


    def take_bias(self, n=1):
        self.set('OBJECT', 'Bias')
        self.set('OBSTYPE', 'Bias')
        self.set('AUTOSHUT', False)
        self.set_exptime(0)
        self.goi(n=n)
        self.set('AUTOSHUT', True)


    def take_dark(self, exptime, n=1):
        self.set('OBJECT', 'Dark')
        self.set('OBSTYPE', 'Dark')
        self.set('AUTOSHUT', False)
        self.set_exptime(exptime)
        self.goi(n=n)
        self.set('AUTOSHUT', True)


    def take_flat(self, exptime, lamp='quartz1', lampfilter='ug5', slit='B2', n=1):
        self.set('OBJECT', 'Flat')
        self.set('OBSTYPE', 'Flat')
        self.set('AUTOSHUT', True)
        self.set_exptime(exptime)
        self.open_covers()
        self.set_lamp(lamp)
        self.set_lamp_filter(lampfilter)
        self.set_filter1('clear')
        self.set_filter2('clear')
        self.set_slit(slit)
        self.goi(n=n)
        self.set_lamp('none')


class HIRESr(HIRES):
    def __init__(self, logger=None):
        self.logger = logger
        super(HIRESr, self).__init__(logger=self.logger, mode='Red')


class HIRESb(HIRES):
    def __init__(self, logger=None):
        self.logger = logger
        super(HIRESb, self).__init__(logger=self.logger, mode='Blue')
