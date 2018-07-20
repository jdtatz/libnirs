"""
Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.

Michał Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
Marta Łukowska (Jagiellonian University)


The module is offline Python implementation of QRS complex detection in the ECG signal based
on the Pan-Tomkins algorithm: Pan J, Tompkins W.J., A real-time QRS detection algorithm,
IEEE Transactions on Biomedical Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236.

The QRS complex corresponds to the depolarization of the right and left ventricles of the human heart. It is the most visually obvious part of the ECG signal. QRS complex detection is essential for time-domain ECG signal analyses, namely heart rate variability. It makes it possible to compute inter-beat interval (RR interval) values that correspond to the time between two consecutive R peaks. Thus, a QRS complex detector is an ECG-based heart contraction detector.

Offline version detects QRS complexes in a pre-recorded ECG signal dataset (e.g. stored in .csv format).

This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. It was created and used for experimental purposes in psychophysiology and psychology.

You can find more information in module documentation:
https://github.com/c-labpl/qrs_detector

If you use these modules in a research project, please consider citing it:
https://zenodo.org/record/583770

If you use these modules in any other project, please refer to MIT open-source license.


MIT License

Copyright (c) 2017 Michał Sznajder, Marta Łukowska

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from scipy.signal import butter, lfilter


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def findpeaks(data, spacing=1, limit=None):
    """
    Janko Slavic peak detection algorithm and implementation.
    https://github.com/jankoslavic/py-tools/tree/master/findpeaks
    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param ndarray data: data
    :param int spacing: minimum spacing to the next peak
    :param float limit: peaks should have value greater or equal
    :return array: detected peaks indexes array
    """
    length = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + length] = data
    peak_candidate = np.zeros(length)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + length]  # before
        start = spacing
        h_c = x[start: start + length]  # central
        start = spacing + s + 1
        h_a = x[start: start + length]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def detect_qrs(ecg_data_raw, plot_axes=None, *, signal_frequency=250, filter_lowcut=1e-6, filter_highcut=15/0.72, filter_order=1, integration_window=9, findpeaks_limit=.5*0.35, findpeaks_spacing=36, refractory_period=86, qrs_peak_filtering_factor=0.125, noise_peak_filtering_factor=0.125, qrs_noise_diff_weight=0.25):
    # Measured and calculated values.
    qrs_peak_value = 0.0
    noise_peak_value = 0.0
    threshold_value = 0.0

    # Detection results.
    qrs_peaks_indices = []
    noise_peaks_indices = []

    # Run whole detector flow.

    # Detect Peaks
    # Extract measurements from loaded ECG data.
    ecg_measurements = ecg_data_raw[:, 1]
    # Measurements filtering - 0-15 Hz band pass filter.
    filtered_ecg_measurements = bandpass_filter(ecg_measurements, filter_lowcut, filter_highcut, signal_frequency, filter_order)
    filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
    # Derivative - provides QRS slope information.
    differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)
    # Squaring - intensifies values received in derivative.
    squared_ecg_measurements = differentiated_ecg_measurements ** 2
    # Moving-window integration.
    integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))
    # Fiducial mark - peak detection on integrated measurements.
    detected_peaks_indices = findpeaks(integrated_ecg_measurements, findpeaks_spacing, findpeaks_limit)
    detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

    # Detect QRS
    last_qrs_index = 0
    for detected_peak_index, detected_peaks_value in zip(detected_peaks_indices, detected_peaks_values):
        # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
        if detected_peak_index - last_qrs_index > refractory_period or 0 == len(qrs_peaks_indices):
            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically set threshold value.
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices.append(detected_peak_index)
                last_qrs_index = detected_peak_index

                # Adjust QRS peak value used later for setting QRS-noise threshold.
                qrs_peak_value = qrs_peak_filtering_factor * detected_peaks_value + (1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices.append(detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise threshold.
                noise_peak_value = noise_peak_filtering_factor * detected_peaks_value + (1 - noise_peak_filtering_factor) * noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            threshold_value = noise_peak_value + qrs_noise_diff_weight * (qrs_peak_value - noise_peak_value)

    # Create array containing both input ECG measurements data and QRS detection indication column.
    # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
    measurement_qrs_detection_flag = np.zeros([len(ecg_data_raw[:, 1]), 1])
    measurement_qrs_detection_flag[qrs_peaks_indices] = 1
    ecg_data_detected = np.append(ecg_data_raw, measurement_qrs_detection_flag, 1)
    
    if plot_axes is not None:
        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices):
            axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)
        
        plot_data(axis=plot_axes[0], data=ecg_data_raw[:, 1], title='Raw ECG measurements')
        plot_data(axis=plot_axes[1], data=filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=plot_axes[2], data=differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=plot_axes[3], data=squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=plot_axes[4], data=integrated_ecg_measurements, title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=plot_axes[4], values=integrated_ecg_measurements, indices=qrs_peaks_indices)
        plot_data(axis=plot_axes[5], data=ecg_data_detected[:, 1], title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=plot_axes[5], values=ecg_data_detected[:, 1], indices=qrs_peaks_indices)
    
    qrs_peaks_indices = np.asarray(qrs_peaks_indices)
    qrs_peaks_vals = ecg_data_raw[qrs_peaks_indices, 0]
    return qrs_peaks_indices, qrs_peaks_vals
