#!/usr/env/python

from __future__ import division, print_function

## Import General Tools
import sys
import os
import argparse
import logging
import re
from time import sleep
from ast import literal_eval

from datetime import datetime as time
from datetime import timedelta as dt

import numpy as np

import ktl
from ktl.Exceptions import ktlError

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
                try:
                    result = (self.services[name]).read(kw)
                    kwfound = True
                except ktlError as e:
                    self.warning(e)
                    self.warning('Trying again')
                    try:
                        result = (self.services[name]).read(kw)
                        kwfound = True
                    except ktlError as e:
                        self.error(e)
                        raise
                break
        if not kwfound:
            raise InstrumentError('{} not in {} related services'.format(
                  kw, self.name))
        return result


    def set(self, kw, val):
        kwfound = False
        for name in self.services.keys():
            if kw in self.services[name].keywords():
                try:
                    (self.services[name]).write(kw, val)
                    kwfound = True
                except ktlError as e:
                    self.warning(e)
                    self.warning('Trying again')
                    try:
                        (self.services[name]).write(kw, val)
                        kwfound = True
                    except ktlError as e:
                        self.error(e)
                        raise
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
        self.lamps = ['none', 'off', 'thar1', 'thar2', 'quartz1', 'quartz2']
        self.lamp_filters = ['dt', 'ug1', 'gg495', 'bg12', 'bg14', 'ng3',
                             'clear', 'ug5', 'etalon', 'bg13', 'bg38']
        self.filters1 = ['bg24a', 'kv408', 'kv370', 'kv389', 'clear', 'rg610',
                         'og530', 'kv418', 'kv380', 'gg475', 'wg335', 'wg360']
        self.filters2 = ['cuso4', '6563/30', 'dt', 'clear', '5026/600', '6300/30',
                         'home', '3090/62', '6199/30', '5893/30']
#         self.xdangle_range = []
#         self.echangle_range = []

        self.get_services()
        self.state = []
        self.previous_state = []
        ## Expressions for state of covers
        self.col_cover_o = ktl.Expression("$hires.{0}CCVCLOS == 'not closed' and $hires.{0}CCVOPEN == 'opened'".format(self.mode[0]))
        self.col_cover_c = ktl.Expression("$hires.{0}CCVCLOS == 'closed' and $hires.{0}CCVOPEN == 'not open'".format(self.mode[0]))
        self.ech_cover_o = ktl.Expression("$hires.ECOVCLOS == 'not closed' and $hires.ECOVOPEN == 'opened'")
        self.ech_cover_c = ktl.Expression("$hires.ECOVCLOS == 'closed' and $hires.ECOVOPEN == 'not open'")
        self.xd_cover_o = ktl.Expression("$hires.XCOVCLOS == 'not closed' and $hires.XCOVOPEN == 'opened'")
        self.xd_cover_c = ktl.Expression("$hires.XCOVCLOS == 'closed' and $hires.XCOVOPEN == 'not open'")
        self.c1_cover_o = ktl.Expression("$hires.C1CVCLOS == 'not closed' and $hires.C1CVOPEN == 'opened'")
        self.c1_cover_c = ktl.Expression("$hires.C1CVCLOS == 'closed' and $hires.C1CVOPEN == 'not open'")
        self.c2_cover_o = ktl.Expression("$hires.C2CVCLOS == 'not closed' and $hires.C2CVOPEN == 'opened'")
        self.c2_cover_c = ktl.Expression("$hires.C2CVCLOS == 'closed' and $hires.C2CVOPEN == 'not open'")
        self.cam_cover_o = ktl.Expression("$hires.CACVCLOS == 'not closed' and $hires.CACVOPEN == 'opened'")
        self.cam_cover_c = ktl.Expression("$hires.CACVCLOS == 'closed' and $hires.CACVOPEN == 'not open'")
        self.dks_cover_o = ktl.Expression("$hires.DARKCLOS == 'not closed' and $hires.DARKOPEN == 'open'")
        self.dks_cover_c = ktl.Expression("$hires.DARKCLOS == 'closed' and $hires.DARKOPEN == 'not open'")

        self.info('Instantiated HIRES in {} mode'.format(self.mode))


    def get_binning(self):
        response = self.get('BINNING')
        Xmatch = re.search('Xbinning\s(\d+)', response)
        Ymatch = re.search('Ybinning\s(\d+)', response)
        if Xmatch and Ymatch:
            xbin = int(Xmatch.group(1))
            ybin = int(Ymatch.group(1))
            return xbin, ybin
        else:
            return None, None


    def print_state_change(self, tick=None):
        if tick is None:
            tick = time.now()
        self.state = [int(literal_eval(self.get('OBSERVIP').title())),
                      int(literal_eval(self.get('EXPOSIP').title())),
#                       int(literal_eval(self.get('HDRCOLIP').title())),
                      int(literal_eval(self.get('ERASEIP').title())),
                      int(literal_eval(self.get('WCRATE').title())),
                      int(literal_eval(self.get('WDISK').title()))]
        if self.state != self.previous_state:
            tock = time.now()
            elapsed = (tock-tick).total_seconds()
            if self.state == [1, 1, 0, 0, 0]:
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Exposing CCD'))
            elif self.state == [1, 0, 0, 1, 0]:
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Reading Out'))
            elif self.state == [1, 0, 0, 0, 0]:
                pass
#                 self.debug('  {:.1f}s: {}'.format(elapsed, 'Exposure in Progress'))
            elif self.state == [1, 0, 0, 0, 1]:
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Writing to Disk'))
            elif self.state == [0, 0, 0, 0, 1]:
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Writing to Disk'))
            elif self.state == [0, 0, 0, 0, 0]:
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Done'))
            else:
                self.debug('  {:.1f}s: {}'.format(elapsed, self.state))
        self.previous_state = self.state


    def goi(self, n=1):
        images = []
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
                tock = time.now()
                elapsed = (tock-tick).total_seconds()
                self.debug('  {:.1f}s: {}'.format(elapsed, 'Erasing CCD'))
                done = ktl.Expression('($hiccd.OBSERVIP == false) and ($hiccd.WDISK == false)')
                self.set('EXPOSE', True)
                while not done.evaluate():
                    self.print_state_change(tick=tick)
                    sleep(0.1)
                sleep(1.0) # time shim :(
                tock = time.now()
                elapsed = (tock-tick).total_seconds()
                self.info('  File will be written to: {}'.format(outfile))
                images.append(outfile)
                self.info('  Done ({:.1f} s elapsed)'.format(elapsed))
        return images

    def all_open(self):
        return (self.col_cover_o.evaluate()
                and self.ech_cover_o.evaluate()
                and self.xd_cover_o.evaluate()
                and self.c1_cover_o.evaluate()
                and self.c2_cover_o.evaluate()
                and self.cam_cover_o.evaluate()
                and self.dks_cover_o.evaluate()
                and not self.col_cover_c.evaluate()
                and not self.ech_cover_c.evaluate()
                and not self.xd_cover_c.evaluate()
                and not self.c1_cover_c.evaluate()
                and not self.c2_cover_c.evaluate()
                and not self.cam_cover_c.evaluate()
                and not self.dks_cover_c.evaluate())


    def all_closed(self):
        return (self.col_cover_c.evaluate()
                and self.ech_cover_c.evaluate()
                and self.xd_cover_c.evaluate()
                and self.c1_cover_c.evaluate()
                and self.c2_cover_c.evaluate()
                and self.cam_cover_c.evaluate()
                and self.dks_cover_c.evaluate()
                and not self.col_cover_o.evaluate()
                and not self.ech_cover_o.evaluate()
                and not self.xd_cover_o.evaluate()
                and not self.c1_cover_o.evaluate()
                and not self.c2_cover_o.evaluate()
                and not self.cam_cover_o.evaluate()
                and not self.dks_cover_o.evaluate())


    def open_covers(self):
        if not self.all_open():
            tick = time.now()
            self.info('Opening {} covers'.format(self.mode))
            self.set('{}COCOVER'.format(self.mode[0].upper()), 'open')
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
        if not self.all_closed():
            tick = time.now()
            self.info('Closing {} covers'.format(self.mode))
            self.set('{}COCOVER'.format(self.mode[0].upper()), 'closed')
            self.set('ECHCOVER', 'closed')
            self.set('XDCOVER', 'closed')
            self.set('CO1COVER', 'closed')
            self.set('CO2COVER', 'closed')
            self.set('CAMCOVER', 'closed')
            self.set('DARKSLID', 'closed')
            tock = time.now()
            elapsed = (tock-tick).total_seconds()
            self.info('  Done ({:.1f} s elapsed)'.format(elapsed))


    def set_slit(self, slit):
        assert slit.upper() in self.slits
        self.set('DECKNAME', slit.upper())


    def set_lamp(self, lamp):
        assert lamp.lower() in self.lamps
        if lamp.lower() == 'off':
            lamp = 'none'
        self.set('LAMPNAME', lamp)


    def set_lamp_filter(self, lampfilter):
        assert lampfilter in self.lamp_filters
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


    def set_xdangle(self, angle):
#         assert (float(angle) >= self.xdangle_range[0]) and (float(angle) <= self.xdangle_range[1])
        self.set('XDANGLE', float(angle))


    def set_echangle(self, angle):
#         assert (float(angle) >= self.echangle_range[0]) and (float(angle) <= self.echangle_range[1])
        self.set('ECHANGLE', float(angle))


    def take_bias(self, n=1):
        self.set('OBJECT', 'Bias')
        self.set('OBSTYPE', 'Bias')
        self.set('AUTOSHUT', False)
        self.set_exptime(0)
        images = self.goi(n=n)
        self.set('AUTOSHUT', True)
        return images


    def take_dark(self, exptime, n=1):
        self.set('OBJECT', 'Dark')
        self.set('OBSTYPE', 'Dark')
        self.set('AUTOSHUT', False)
        self.set_exptime(exptime)
        images = self.goi(n=n)
        self.set('AUTOSHUT', True)
        return images


    def take_flat(self, exptime, lamp='quartz1', lampfilter='ug5', slit='B2', n=1):
        self.set('OBJECT', 'Flat')
        self.set('OBSTYPE', 'Flat')
        self.set('AUTOSHUT', True)
        self.set('LMIRR', 'in')
        self.set_exptime(exptime)
        self.open_covers()
        self.set_lamp(lamp)
        self.set_lamp_filter(lampfilter)
        self.set_filter1('clear')
        self.set_filter2('clear')
        self.set_slit(slit)
        images = self.goi(n=n)
        self.set('LMIRR', 'out')
        self.set_lamp('none')
        return images


class HIRESr(HIRES):
    def __init__(self, logger=None):
        self.logger = logger
        super(HIRESr, self).__init__(logger=self.logger, mode='Red')


class HIRESb(HIRES):
    def __init__(self, logger=None):
        self.logger = logger
        super(HIRESb, self).__init__(logger=self.logger, mode='Blue')
