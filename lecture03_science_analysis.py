"""
3 Science Analysis
"""
import pickle

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# =============================================================================
print("\n\n 3.1. Read the data")
print("=============================================================================\n")

"""
3.1. Read the data
"""

taste_dir = 'TASTE_analysis/group05_QATAR-1_20230212'

science_list = np.genfromtxt(f'{taste_dir}/science/science.list', dtype='str')
print(science_list)

# taking a subset of images
science_test_list = science_list[:10]

# =============================================================================
print("\n\n 3.2. Data reduction steps")
print("=============================================================================\n")

"""
3.2. Data reduction steps

Data reduction steps
Science data reduction includes these steps:

1 multiplication by gain
2 bias subtraction
3 division by flat

These components contribute to the error budget:

- readout noise
- error associated with the bias
- error associated with the flat
- photon noise

We load the median bias and the median normalized flat (with its error)
We must use the same bias error and readout noise estimates we used for the flat correction.
"""

median_bias = pickle.load(open(f"{taste_dir}/bias/median_bias.p", "rb"))
bias_std = 1.3  # [e] photoelectrons
readout_noise = 7.4  # [e] photoelectrons
gain = 1.91  # [e/ADU]  # this value may be
median_normalized_flat = pickle.load(open(f"{taste_dir}/flat/median_normalized_flat.p", "rb"))
median_normalized_flat_errors = pickle.load(open(f"{taste_dir}/flat/median_normalized_flat_errors.p", "rb"))

"""
For the photon noise, we can rely on Poissonian distribution as done with the flat. Remember that the photon 
noise must be computed after removing the bias but before correcting for the flat field, as the error must 
be calculated on the actual number of photons received on the detector, not on the photons emitted by the source.
"""

"""
Do not worry if you run into this warning, as it is likely due to the presence of zero values in the overscan region of the flat:
RuntimeWarning: divide by zero encountered in divide

"""

# suppress divide by zero warning
np.seterr(divide='ignore', invalid='ignore')


for science_name in science_test_list:
    science_fits = fits.open(f'{taste_dir}/science/' + science_name)
    science_data = science_fits[0].data * gain  # save the data from the first HDU
    science_fits.close()

    science_debiased = science_data - median_bias
    science_corrected = science_debiased / median_normalized_flat

    ## Error associated to the science corrected frame
    science_debiased_errors = np.sqrt(readout_noise ** 2 + bias_std ** 2 + science_debiased)
    science_corrected_errors = science_corrected * np.sqrt((science_debiased_errors / science_debiased) ** 2 + (
            median_normalized_flat_errors / median_normalized_flat) ** 2)

# =============================================================================
print("\n\n 3.3. Save the images")
print("=============================================================================\n")

"""
3.3. Save the images

We now repeat the same steps again, but this time, we save the output into a file.
First of all, from the terminal let's make a directory where to store the new files. \

mkdir correct
Be sure to be in the same folder where you launched the Jupyter Notebbok.

We want to use the same names of the files for our new frames, without the .fits extension. 
The extension is made by five characters, i.e., we can take the string identifying each file name 
and remove the last five characters. In Python notation:

science_name[:-5]
The new file name is made by adding new string characters at the beginning and at the end of the original name:

new_name = './correct/' + science_name[:-5] + '_corr.p'
"""

for science_name in science_test_list:
    science_fits = fits.open(f'{taste_dir}/science/' + science_name)
    science_data = science_fits[0].data * gain  # save the data from the first HDU
    science_fits.close()

    science_debiased = science_data - median_bias
    science_corrected = science_debiased / median_normalized_flat

    ## Error associated to the science corrected frame
    science_debiased_errors = np.sqrt(readout_noise ** 2 + bias_std ** 2 + science_debiased)
    science_corrected_errors = science_corrected * np.sqrt((science_debiased_errors / science_debiased) ** 2 + (
            median_normalized_flat_errors / median_normalized_flat) ** 2)

    new_name = f'{taste_dir}/correct/' + science_name[:-5] + '_corr.p'
    # noinspection PyTypeChecker
    pickle.dump(science_corrected, open(new_name, 'wb'))
    new_name = f'{taste_dir}/correct/' + science_name[:-5] + '_corr_errors.p'
    # noinspection PyTypeChecker
    pickle.dump(science_corrected_errors, open(new_name, 'wb'))


# =============================================================================
print("\n\n 3.4. Extracting and saving useful information")
print("=============================================================================\n")

"""
Extracting and saving useful information
For our analysis, we need to know when the data has been taken, not just how. In addition, 
we may need some extra information regarding the pointing of the telescope.

The epoch of the exposure, expressed in Julian date
The exposure time, i.e., the duration of our exposure
The airmass of the telescope during the exposure
We extract this information from the header of each fits file. Other info you may need, e.g., the filter used 
for the observations, do not change with time so you just need to extract them from a single frame.
"""

n_images = len(science_test_list)

array_jd = np.zeros(n_images)
array_exptime = np.zeros(n_images)
array_airmass = np.zeros(n_images)

for i_science, science_name in enumerate(science_test_list):

    science_fits = fits.open(f'{taste_dir}/science/' + science_name)
    array_jd[i_science] = science_fits[0].header['JD']
    array_exptime[i_science] = science_fits[0].header['EXPTIME']
    array_airmass[i_science] = science_fits[0].header['AIRMASS']

    ## Let's print the comment for the first image
    if i_science == 0:
        print('', science_fits[0].header.comments['JD'])
        print('', science_fits[0].header.comments['EXPTIME'])
        print('', science_fits[0].header.comments['AIRMASS'])

    science_fits.close()

# =============================================================================
print("\n\n 3.5. Conversion to BJD_TDB")
print("=============================================================================\n")

"""
Conversion to BJD_TDB
Did you notice that the Julian Date is expressed as JD and not BJD, i.e., Barycentric Julian Date? This is the 
time when an event is recorded at the observatory. What we want in reality is the time as recorded at the Barycenter 
of the Solar System, as we do not want to be influenced by the specific position of the observatory when the 
observations are taken.

Check the story of how the Danish astronomer Ole Roemer (1644–1710) became the first person to measure the speed 
of light in 1676, to see how the position of the observer may influence your results.

JD and BJD are independent of the time standard, i.e., the way we measure the time between two events 
at the same location,

The correction is done through several steps:

We want each time stamp to be associated with the centre of the exposure, not the initial moment. 
We have to shift all the time stamps by half the exposure time
We have to shift from Coordinated Universal Time (UTC) (a discontinuous scale due to the introduction of leap seconds) 
to the Barycentric Dynamical Time (TDB), a relativistic coordinate time scale defined on the barycenter of the Solar System.
We have to correct for the light travel time effect, i.e.,  the time required by the light to travel between Earth 
(specifically, our observatory) and the Solar System barycenter
We rely on the Time package of Astropy to perform all these calculations. Check the documentation for more examples.

First of all, we need to specify an Astronomical Coordinate System for our target as required by the Astropy package. 
For this purpose, we rely on the coordinates package.

We also need to specift the location of the observatory. They can be expressed as geocentric of geodetic coordinates, 
by following the instructions provided in EarthLocation. The location of the observatory 
is provided as an argument when calling the Time class.
"""

from astropy import coordinates as coord, units as u
target = coord.SkyCoord("20:13:31.61","+65:09:43.49",unit=(u.hourangle, u.deg), frame='icrs')

## install jplephem

from astropy.time import Time
#https://docs.astropy.org/en/stable/time/

# let's compute the light travel time for one year of observations
jd_plot = np.arange(2460000, 2460365.25, 0.10)
tm_plot = Time(jd_plot, format='jd', scale='utc', location=('45.8472d', '11.569d'))
# questo la prima volta va lento perchè si deve scaricare tutti i dati
ltt_plot = tm_plot.light_travel_time(target, ephemeris='jpl')

# Convert to BJD_TDB, and then add the light travel time
bjd_tdb_plot = tm_plot.tdb + ltt_plot

plt.figure(figsize=(6,4))
plt.plot(jd_plot, ltt_plot.to_value(u.min))
plt.xlabel('JD [d]')
plt.ylabel('Light travel time [m]')
plt.show()

# =============================================================================
print("\n\n 3.6. Information regarding the Light Travel Time calculation")
print("=============================================================================\n")

"""
Information regarding the Light Travel Time calculation
Link to the method: light_travel_time

light_travel_time(skycoord, kind='barycentric', location=None, ephemeris=None)
Light travel time correction to the barycentre or heliocentre.

The frame transformations used to calculate the location of the solar system barycentre and the 
heliocentre rely on the erfa routine epv00, which is consistent with the JPL DE405 ephemeris to an accuracy of 11.2 km, 
corresponding to a light travel time of 4 microseconds.

The routine assumes the source(s) are at large distance, i.e., neglects finite-distance effects.

Returns: time_offset TimeDelta
The time offset between the barycentre or Heliocentre and Earth, in TDB seconds. Should be added to the original 
time to get the time in the Solar system barycentre or the Heliocentre. Also, the time conversion to BJD will 
then include the relativistic correction as well.

Below, we compute the JD_UTC at mid exposure before converting to BJD_TDB
"""

jd = array_jd + array_exptime/86400./2.

tm = Time(jd, format='jd', scale='utc', location=('45.8472d', '11.569d'))

# Asiago - Cima Ekar
# 45° 50' 50'' N -> 45.8472
# 11° 34' 08'' E -> 11.569

ltt_bary = tm.light_travel_time(target)

bjd_tdb = tm.tdb + ltt_bary

print('Average Light travel time:                     {0:12.2f} minutes'.format(np.average(ltt_bary.to_value(u.min))))
print('Average difference between JD_UTC and BJD_TDB: {0:12.2f} seconds'.format(np.average(jd - bjd_tdb.to_value('jd'))*86400))
