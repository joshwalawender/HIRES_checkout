#!/bin/sh

# Script for taking HRES data for detector characterization
#    GD  19 April 2016

# Objective: To measure and track HIRES detector characteristics
#             (i.e. Readnoise, Dark current, and Gain)

set datadir = `show -s hires -terse outdir`

echo "Data will be saved to this directory: $datadir"

#set some other header values
modify -s hiccd observer='Hires Master'
modify -s hiccd frameno=1

############################
# Take a series of biases
############################

modify -s hiccd object='Bias'
modify -s hiccd obstype=Bias

# set the shutter to stay closed
modify -s hiccd autoshut=f

set num_biases = 10
modify -s hiccd ttime=0
goi $num_biases

############################
# Take a series of darks
############################

set num_darks = 5

modify -s hiccd object='Dark'
modify -s hiccd obstype=Dark

# set the shutter to stay closed
modify -s hiccd autoshut=f

set times = ( 6 60 600 )

foreach itime ($times)

  # put in a divider between integration times
  echo "-----------------------------------------------------"
  echo "Dark: itime = $itime"
  echo "-----------------------------------------------------"

  modify -s hiccd ttime=$itime
  goi $num_darks

  end

# end foreach loop

############################
# Take a series of flats
############################

modify -s hiccd object='Flat'
modify -s hiccd obstype=Flat

# set the shutter to open during expsoures
modify -s hiccd autoshut=t

# need to find a test for whether we have blue or red collimator in place

# open the covers for the blue mode
#open.blue

# or open the covers for the red mode
open.red

set num_flats = 5

# turn on the quartz1 lamps and put in the mirror

modify -s hires lampname=quartz1

# put in the appropriate lamp filter
modify -s hires lfilname='ug5'

# make sure the spectroscopic filters are both blank
modify -s hires fil1name='clear'
modify -s hires fil2name='clear'

# select the slit
modify -s hires deckname='B2'

# set itimes to: 1,5,10,20 sec

# set the shutter to open when taking an exposure
modify -s hiccd autoshut=t

set times = ( 1 5 10 20 )

foreach itime ($times)

  # put in a divider between integration times
  echo "-----------------------------------------------------"
  echo "Flat: itime = $itime"
  echo "-----------------------------------------------------"

  modify -s hiccd ttime=$itime
  goi $num_flats

  end

# end foreach loop

# turn lamp off and stow the mirror
modify -s hires lampname='none'



# shut down all the HIRES software
stop_all_hires_guis

  echo "-----------------------------------------------------"
  echo "-----------------------------------------------------"
  echo "Script Done. Shutting down HIRES software"
  echo "Data saved: $datadir"

exit

