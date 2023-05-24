# import packages
import numpy as np
import matplotlib.pyplot as plt

import mne

from autoreject import AutoReject

# ------------------------------------------------------------
# Read the data from the .set file
# The sample data is recorded using 32-channels EEG system
# Data was recorded during a picture naming task
# Sampling frequency: 1024 Hz
# Electrode montage: 10-20 international system

path = 'test-data/Sets/'
raw = mne.io.read_raw_eeglab(path + '10_2342019_2.set', preload=True)

# Plot the raw data
raw.plot()

# Plot the sensor locations
raw.set_montage('standard_1020')
raw.plot_sensors(kind='topomap', show_names=True)
             
# ------------------------------------------------------------
# Preprocessing
# -------------
# The following preprocessing steps are a non-exhaustive list of steps that one can apply to the data. They are used just for the sake of the tutorial.
# -------------

# Drop the first 50 seconds of data + stop at 550 seconds
raw.crop(tmin=50, tmax=550)

# Filter the data 
# Notch filter
# raw.notch_filter(50)

# Bandpass filter the data from 1 - 45 Hz
filt_raw = raw.copy().filter(l_freq=1, h_freq=45)

# Plot filtered data
filt_raw.plot()

# Run Independent Component Analysis (ICA) to remmove artifacts 
ica = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter='auto')
ica.fit(filt_raw)

# Plot the components on the topomap
ica.plot_components()
filt_raw.load_data()
# plot the signal of the component
ica.plot_sources(filt_raw)

# After visual inspection, we will remove components 10 and 18
ica.exclude = [0,1,6]
filt_raw.load_data()
ica.apply(filt_raw)

# Plot the data after removing the IC
filt_raw.plot()

# ------------------------------------------------------------
# Epoching
# --------
# Epoch the data based on the event number 33273
# The event number 33273 is the event number for the picture naming task = it is the event number for the onset of the picture

events, event_id = mne.events_from_annotations(raw)
id = event_id['33273']
epochs = mne.Epochs(filt_raw, events, id, tmin=-0.2, tmax=0.4, baseline=(None, 0), preload=True, event_repeated='merge')


evoked = epochs.average()
evoked.plot()
# ------------------------------------------------------------
# Continue preprocessing
# ----------------------
# Run AutoReject to reject bad epochs and repair bad channels
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)

# Plot the epochs
epochs_clean.plot()

# Plot one sensor in all epochs
epochs_clean.plot_image(picks=['Oz'])

evoked_clean = epochs_clean.average()
evoked_clean.plot()
# ------------------------------------------------------------
# Frequency analysis
# ------------------

# Compute and plot the PSD for each channel
epochs_clean.compute_psd(fmin=1, fmax=45).plot(average=False)

# Compute and plot the PSD averaged across all channels
epochs_clean.compute_psd(fmin=1, fmax=45).plot(average=True, color='blue')

# Compute and plot topomap of PSD
epochs_clean.compute_psd(fmin=1, fmax=45).plot_topomap(normalize=False, contours=0)

# ------------------------------------------------------------
# Time-frequency analysis
# -----------------------
# Define frequency of interest (log-spaced)
freqs = np.logspace(*np.log10([6, 35]), num=10)
n_cycles = 2.0   

# Compute power coherence using Morlet wavelets
power, itc = mne.time_frequency.tfr_morlet(epochs_clean, freqs=freqs, n_cycles=n_cycles, return_itc=True, average=True, decim=3)

# power.apply_baseline(baseline=(-0.2, 0), mode='logratio')

# Plot topomaps
# The following figure is an interactive figure. You can click on the topomap to see the TFR for a specific channel.
power.plot_topo(baseline=(-0.2,0), mode='logratio', title='Average power')

# Plot TFR for one channel
power.plot([5], title=power.ch_names[5], baseline=(-0.2,0), mode='logratio')

# Plot aggregated TFR across topomaps at specific times and frequencies
power.plot_joint(baseline=(-0.2,0), tmin=-0.5, tmax=0.21, timefreqs=[(-0.1,10),(0.2, 10)])