
import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Anomaly detection

def exp_mean_avg(x, alpha):
    m = np.zeros((1, x.shape[-1]))
    y = np.array(x)
    for i in range(x.shape[0]):
        m = alpha * x[i,:] + (1 - alpha) * m
        y[i,:] = m
    return y

def id_from_timestamp(ts, ts_bounds, max_id):
    ts_min, ts_max = ts_bounds
    return int((ts - ts_min) / (ts_max - ts_min) * max_id)

def sum_conv_error(error, conv = None):
    err_sum = np.sum(error, axis=1)
    if conv != None:
        err_sum = np.convolve(err_sum, conv, mode="same")
    return err_sum

def conv_error(error, conv = None):
    if conv != None:
        error = np.convolve(error, conv, mode="same")
    return error


## Gaussian method

def gaussSum(x,*p):
    n=int(len(p)/3)
    A=p[:n]
    w=p[n:2*n]
    c=p[2*n:3*n]
    y = sum([ A[i]*np.exp(-(x-c[i])**2./(2.*(w[i])**2.))/(2*np.pi*w[i]**2)**0.5 for i in range(n)])
    return y

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def build_gauss_error_model(error, nbuckets = 100, display = False):
    mean_error = np.mean(error)
    sd_error = np.sqrt(np.mean((error - mean_error)**2))
    max_error = np.max(error)
    min_error = np.min(error)

    if display:
        print("mean error: " + str(mean_error))
        print("max error: " + str(max_error))
        print("min error: " + str(min_error))
        print("sd error: " + str(sd_error))

    hist, bin_edges = np.histogram(error, nbuckets, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    default_A = max(hist)

    p0 = [default_A, mean_error, sd_error]
    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0, method='trf')

    if display:
        print('fitted mean = ' + str(coeff[1]))
        print('fitted stddev = ' + str(coeff[2]))

    return coeff, bin_centres, hist

def get_error_distribution(error, nbuckets = 100, display = False):
    mean_error = np.mean(error)
    sd_error = np.sqrt(np.mean((error - mean_error)**2))
    max_error = np.max(error)
    min_error = np.min(error)

    if display:
        print("mean error: " + str(mean_error))
        print("max error: " + str(max_error))
        print("min error: " + str(min_error))
        print("sd error: " + str(sd_error))

    hist, bin_edges = np.histogram(error, nbuckets, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    default_A = max(hist)
    coeff = (default_A, mean_error, sd_error)

    return coeff, bin_centres, hist

def display_error_distribution(bins, hist, coeff, title = "", xmin = -10, xmax = 10):
    plt.title(title)
    plt.xlabel("error amplitude")
    plt.ylabel("proportion of elements")

    plt.plot(bins, hist, label='measured error')
    hist_fit = gauss(bins, *coeff)
    stddev = coeff[2]
    plt.plot(bins, hist_fit, label='modeled error')
    plt.xlim((xmin * stddev ,xmax * stddev))

    plt.legend()
    plt.show()

class ErrorNormalizationModel:
    def __init__(self):
        self.coeffs = []

    def fit(self, error, conv = None, fit_gauss = False, display = False):
        self.coeffs = []

        for i in range(error.shape[1]):
            error_i = error[:,i]
            error_i = conv_error(error_i, conv)
            if fit_gauss:
                coeff, bins, hist = build_gauss_error_model(error_i, display = display)
            else:
                coeff, bins, hist = get_error_distribution(error_i, display = display)
            _, mean, stddev = coeff

            if display:
                display_error_distribution(bins, hist, coeff, "error distribution")

            self.coeffs.append((mean, stddev))

    def anomaly_dates(self, error, stddev_threshold = 3, conv = None, display = False):
        error_dev = np.zeros(error.shape[0])
        total = 0

        for i in range(error.shape[1]):
            mean, stddev = self.coeffs[i]
            error_i = error[:,i]
            error_i = conv_error(error_i, conv)
            error_dev += np.absolute(error_i - mean) / abs(stddev)
            total += 1

        error_dev /= total
        # error_dev = conv_error(error_dev, conv)
        ano_indices = np.where(error_dev > stddev_threshold)[0]

        if display:
            print("Selected %0.1f%% of errors" % (len(ano_indices) / len(error_dev) * 100))
        return ano_indices, error_dev, None

## Vote method

def error_vote(error,nvotes,weight=True, conv = None):
    errorcp = np.copy(error)
    votes = np.zeros(error.shape[0])
    for v in range(nvotes):
        timeindexes = np.argmax(errorcp,axis=0)
        for signal,ti in enumerate(timeindexes):
            errorcp[ti,signal] = 0.0
            if weight:
                votes[ti] = votes[ti]+1*error[ti,signal]
            else:
                votes[ti] = votes[ti]+1
    if conv != None:
        votes = np.convolve(votes, conv, mode="same")
    return votes

def anomaly_dates_votes(votes,nanomalies):
    anomalies_dates = []
    notescp = np.copy(votes)

    for i in range(nanomalies):
        aindex = np.argmax(notescp)
        anomalies_dates.append(aindex)
        notescp[aindex] = 0
    return anomalies_dates


## Peaks method

def error_vote_peaks(error, conv = None, peak_width = 10):
    # TODO vote, but select peaks instead of just the biggest error values
    peaks = []
    for signal in range(error.shape[1]):
        peaks.append(scipy.signal.find_peaks(error[:,signal], width=peak_width))

    err_peaks = np.zeros_like(error)
    for signal in range(len(peaks)):
        for p in peaks[signal][0]:
            err_peaks[p, signal] = error[p, signal]

    err_peaks = sum_conv_error(err_peaks, conv)
    return err_peaks #, peaks

def anomaly_dates_peaks(error, nanomalies, conv = None, peak_width = 10):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    # TODO add coefficient on each signal
    err_sum = sum_conv_error(error, conv)
    peaks, values = scipy.signal.find_peaks(err_sum, width=peak_width)
    values = values["prominences"]
    ano_indices = np.argsort(values)[-nanomalies:]
    return peaks[ano_indices], err_sum, peaks
