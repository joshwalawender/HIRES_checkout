Script to automatically measure HIRES read noise, dark current, and gain.

Script Outline
* Take n bias frames (default n=8)
    * Combine n-1 frames and subtract from the nth frame to make difference frame.
    * Read noise is estimated from the variance in the difference frame.
* Take dark frames at several exposure times
    * Determine dark current by fitting line to median level of bias subtracted darks as a function of exposure time.
* Take flats and determine gain by fitting variance versus flux level.

