import os
import sys
import csv
import io
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.optimize import curve_fit
import tqdm
from tqdm import notebook
import ipywidgets as widgets

from dd_client import DD

# Helper functions

def delete_service(dd, sname):
    try:
        print(dd.delete_service(sname))
        return True
    except Exception as e:
        if e.response.status_code == 404:
            print('service not found', e)
        else:
            raise
        return False

# To dd_client?
def wait_training_over(dd, sname, sleeptime=5):
    while True:
        try:
            res = dd.get_train(sname)
            if res["head"]["status"] != "running":
                return
        except Exception as e:
            print(res)
            raise
        print("Waiting end of training of %s" % sname)
        time.sleep(sleeptime)

def get_col(header,l):
    for i,hl in enumerate(header):
        if hl == l:
            return i
    print(header, l)
    raise IndexError("not found")

def get_datafiles(datadir, prefix = ""):
    """
    Scan directory for all csv files
    prefix: used in recursive call
    """
    datafiles = []

    for fname in os.listdir(datadir):
        fpath = os.path.join(datadir, fname)
        datafile = os.path.join(prefix, fname)

        if os.path.isdir(fpath):
            datafiles += get_datafiles(fpath, datafile)
        elif fname.endswith(".csv"):
            datafiles.append(datafile)

    return datafiles

def dump_data(data, labels, filename):
    file = open(filename,"w")
    datawriter=csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
    datawriter.writerow(labels)
    for ts in range(data.shape[0]):
        datawriter.writerow(data[ts,:])
    file.close()

def load_target(datafile,labels, progress = True):
    csvfile = open(datafile)
    csv_reader = csv.reader(csvfile, delimiter=',')
    header = next(csv_reader)
    col_labels = []
    targets = []
    for l in labels:
        col_labels.append(get_col(header,l))

    for data in (tqdm.notebook.tqdm(csv_reader) if progress else csv_reader):
         targets.append([float(data[i]) for i in col_labels])
    return targets

def load_data(filename):
    file = open(filename,"r")
    datareader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
    header = next(datareader)
    data = []
    for tsdata in datareader:
        data.append(tsdata)
    return np.asarray(data)

def normalize_error(data):
    eps = 1E-5
    max_data = np.amax(data,axis=0)
    min_data = np.min(data,axis=0)
    norm_data=np.divide((data-min_data),(max_data-min_data+eps))
    return norm_data

def normalize_data(pred,targ):
    eps = 1E-5
    max_data = np.amax(targ,axis=0)
    min_data = np.min(targ,axis=0)
    norm_pred= np.divide((pred-min_data),(max_data-min_data+eps))
    norm_targ= np.divide((targ-min_data),(max_data-min_data+eps))
    return norm_pred, norm_targ

def get_signal_mean_error(pred, targ, feat):
    pred_norm, targ_norm = normalize_data(pred[:,feat], targ[:,feat])
    mean_error = np.mean(np.absolute(targ_norm - pred_norm))
    return mean_error



class Timeseries:
    """
    Train and predict timeseries in jupyter.
    """
    def __init__(self, sname, host = "127.0.0.1", port = "8080", proto = 0, api_path = "",
                model = None, models = [], models_dir = "/opt/platform/models/",
                datafiles = [], datadir = "", output_dir = "/temp/predictions/",
                labels = [], offset=50, gpuid = 0,
                autoregressive = False, batch_size = 50, iter_size = 1,
                iterations = 500000, base_lr=0.001, test_interval = 5000):

        self.sname = sname
        self.models = models
        self.models_dir = models_dir
        self.datafiles = datafiles
        self.datadir = datadir
        self.output_dir = output_dir
        self.labels = labels
        self.offset = offset
        self.batch_size = batch_size
        self.gpuid = gpuid
        self.autoregressive = autoregressive

        self.solver_params = {
            "iter_size": iter_size,
            "iterations": iterations,
            "base_lr": base_lr,
            "test_interval": test_interval
        }

        """
        shift: How much the target is being shifted. As models predict
        at different horizons, shift enable to compare the same sections
        of the targets even if the target is shifted by a certain number
        of timesteps.
        """
        self.shift = 0

        # dict {dataset: target}
        self.targs = {}
        # dict { dataset: {model: preds / errors}}
        self.preds = {}
        # signed error
        self.errors = {}

        self.dd = DD(host, port, proto, api_path)

    def get_predict_sname(self):
        return self.sname + "_predict"

    def delete_service(self, predict = False):
        sname = self.get_predict_sname() if predict else self.sname
        return delete_service(self.dd, sname)

    def get_dump_filenames(self, datafile, model):
        model_out_dir = os.path.join(self.output_dir, model)
        dataname = os.path.splitext(datafile)[0]
        if self.autoregressive:
            pred_out_file = os.path.join(model_out_dir, dataname + "_pred_ar.csv")
            err_out_file = os.path.join(model_out_dir, dataname + "_error_ar.csv")
        else:
            pred_out_file = os.path.join(model_out_dir, dataname + "_pred.csv")
            err_out_file = os.path.join(model_out_dir, dataname + "_error.csv")
        return pred_out_file, err_out_file

    def dump_model_preds(self, models = None):
        if not os.path.exists(self.output_dir):
            print("cannot dump predictions, directory %s does not exist" % self.output_dir)
            return

        if models == None:
            models = self.models

        # models, preds, errors, datafiles, datadir, labels, output_dir,
        for model in tqdm.notebook.tqdm(models):
            model_out_dir = os.path.join(self.output_dir, model)

            # print("creating", model_out_dir, "...")
            os.makedirs(model_out_dir, exist_ok = True)

            for datafile in tqdm.notebook.tqdm(self.datafiles):
                pred_out_file, err_out_file = self.get_dump_filenames(datafile, model)

                # needed if "test/"
                try:
                    os.mkdir(os.path.dirname(pred_out_file))
                except FileExistsError:
                    pass

                dump_data(self.preds[datafile][model], self.labels, pred_out_file)
                dump_data(self.errors[datafile][model], self.labels, err_out_file)


    def load_targets(self, labels = None):
        if labels is None:
            labels = self.labels;

        # return dict for each datafile with target
        for datafile in tqdm.notebook.tqdm(self.datafiles):
            targ_file = os.path.join(self.datadir, datafile)
            if not os.path.exists(targ_file):
                print("cannot load target file %s: does not exist" % datafile)
                continue

            self.targs[datafile] = np.array(load_target(targ_file, labels, progress = False))

    def load_preds_errors(self):
        # return dict of dict for each datafile for each model
        for model in self.models:
            model_out_dir = os.path.join(self.output_dir, model)

            if not os.path.exists(model_out_dir):
                print("cannot load predictions for %s: model directory not found" % model)
                continue

            for datafile in tqdm.notebook.tqdm(self.datafiles):
                if datafile not in self.preds:
                    self.preds[datafile] = {}
                    self.errors[datafile] = {}

                pred_out_file, err_out_file = self.get_dump_filenames(datafile, model)

                if not os.path.exists(pred_out_file):
                    print("cannot load predictions file %s" % pred_out_file)
                    continue

                self.preds[datafile][model] = load_data(pred_out_file)
                self.errors[datafile][model] = load_data(err_out_file)

    def get_test_sets(targs):
        """
        Takes dict {dataset: target} as input, and
        returns a list of tuple with the start and
        end of test set in final dataset
        """
        ids_lengths = []
        split_length = 0

        for key in targs:
            if "test/" in key:
                # XXX: for nbeats, target is missing `backcast` values
                # hence wrong computations of the test sets
                l = targs[key].shape[0]
                split_length = max(l, split_length)
                match = re.match(r".*_(\d+)\.csv", key)

                if match:
                    ids_lengths.append((int(match.group(1)), l))

        return [(i * split_length, i * split_length + l) for i, l in ids_lengths]

    # Anomalies: see anomalies.py

    def compute_anomalies(error_norm, error_nabs, method):
        c = 201
        conv = [1 / c] * c
        ano_signal = None
        ano_peaks = None

        if method == "Peaks":
            # ano_signal = error_peaks(error_norm, conv=conv)
            # anomalies = anomaly_dates(ano_signal, 20)
            anomalies, ano_signal, ano_peaks = anomaly_dates_peaks(error_norm, 20,conv=conv)
        elif method == "Votes":
            # Does not work:
            ano_signal = error_vote(error_norm, 40, conv=conv)
            anomalies = anomaly_dates(ano_signal, 40)
        elif method == "Gaussian":
            anomalies, ano_signal, ano_peaks = anomaly_dates_gauss(error_nabs)

        return anomalies, ano_signal, ano_peaks


    # Plot / Widgets

    # TODO Add shift option
    def plot_dataset(self, targ, signals, tstart = None, tend = None, title = None):
        indices = np.arange(tstart, tend)
        if title != None:
            plt.title(title)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        for s in signals:
            plt.plot(indices, targ[tstart:tend,s], label="signal")
        plt.legend()
        plt.gcf().set_facecolor("w")
        plt.show()

    def dataset_ui(self):
        if len(self.targs) == 0:
            self.load_targets()

        dataset_dropdown = widgets.Dropdown(
            options=self.datafiles,
            description='Dataset:'
        )
        label_dropdown = widgets.Dropdown(
            options=self.labels,
            description='Label:'
        )
        start_text = widgets.BoundedIntText(
            min=0,
            max=100000000,
            value=0,
            description='Start:'
        )
        duration_text = widgets.IntText(
            value=-1,
            description='Duration:'
        )
        run_button = widgets.Button(
            description='Update',
            tooltip='Update'
        )
        out = widgets.Output(layout={'border': '1px solid black'})

        def show_ui():
            with out:
                display(dataset_dropdown,
                        label_dropdown,
                        start_text,
                        duration_text,
                        run_button)

        def run_button_action(b):
            out.clear_output()
            show_ui()

            start = start_text.value
            end = start_text.value + duration_text.value
            feat = self.labels.index(label_dropdown.value)
            dset = dataset_dropdown.value

            signame = self.labels[feat]
            targ = self.targs[dset]

            if end >= len(targ) or duration_text.value <= 0:
                end = len(targ)

            with out:
                title = "signal %s from %d to %d" % (signame, start, end)
                self.plot_dataset(targ, [feat], start, end, title)

        run_button.on_click(run_button_action)
        show_ui()
        return out

    def display_compare(self, preds, targ, errors, signals, tstart, tend, title, tests = None, save = False):
        # preds: dict {model: preds}
        # targ: raw target
        # signal: id of the signal (= line to display)
        # shift: start value offset for x axis
        # this method is used to display one specific dataset
        # plt.subplots_adjust(hspace=1)

        indices = np.arange(self.shift + tstart, self.shift + tend)

        nh = 2
        fig = plt.figure(figsize = (len(self.models) * 6, nh * 4))
        axs = fig.subplots(nh, len(self.models))
        # fig.title(title)
        fig.set_facecolor("w")

        for i in range(len(self.models)):
            model = self.models[i]
            pred = preds[model]
            error = errors[model]

            ax1 = axs[0, i] if len(self.models) > 1 else axs[0]
            ax1.set_title(title + "\n" + model)
            ax1.set_xlabel("time")
            ax1.set_ylabel("amplitude")
            for s in signals:
                s_pred = pred[tstart:tend,s]
                ax1.plot(indices[:len(s_pred)], s_pred, label="pred")
                s_targ = targ[tstart:tend,s]
                ax1.plot(indices[:len(s_targ)], s_targ, label="target")
                # ax1.plot(indices, error[tstart:tend,s], label="error")

            if tests:
                for test_start, test_end in tests:
                    ax1.axvspan(test_start + self.shift, test_end + self.shift, facecolor='green', alpha=0.3)
            ax1.legend()

            ax2 = axs[1, i] if len(self.models) > 1 else axs[1]
            ax2.set_title(title + " error\n" + model)
            ax2.set_xlabel("time")
            ax2.set_ylabel("amplitude")

            for s in signals:
                s_error = error[tstart:tend,s]
                ax2.plot(indices[:len(s_error)], s_error, label="error")

            if tests:
                for test_start, test_end in tests:
                    ax2.axvspan(test_start + self.shift, test_end + self.shift, facecolor='green', alpha=0.3)
            ax2.legend()

        plt.tight_layout()

        if save:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()

    def plot_anomalies(models, target, anomalies, signal, tstart, tend, title, shift):
        """
        anomalies: list of anomaly dates
        """
        hwidth = (tend - tstart) / 200
        indices = np.arange(self.shift + tstart, self.shift + tend)
        xstart = tstart + self.shift

        fig = plt.figure(figsize = (len(self.models) * 6, 4))
        axs = fig.subplots(1, len(self.models))
        # fig.title(title)
        fig.set_facecolor("w")

        for i in range(len(self.models)):
            model = self.models[i]
            anomaly, ano_signal, ano_peaks = anomalies[model]
            ax = axs[i] if len(self.models) > 1 else axs

            ax.set_title(title + "\n" + model)
            ax.set_xlabel("time")
            ax.set_ylabel("amplitude")
            ax.plot(indices, target[tstart:tend,signal], label="target")

            # plot anomaly
            # ax.scatter(anomaly, error[anomaly,signal], label = "anomalies", c="red", marker='x')
            for i in anomaly:
                if tstart < i < tend:
                    ax.axvspan(i - hwidth + self.shift, i + hwidth + self.shift, facecolor='red', alpha=0.3)

            if ano_signal is not None:
                ano_signal = ano_signal / max(0.0001, np.max(np.absolute(ano_signal))) * np.max(target)
                ax.plot(indices, ano_signal[tstart:tend], label="mean error")

                if ano_peaks is not None:
                    ano_peaks = ano_peaks[np.logical_and(tstart < ano_peaks, ano_peaks < tend)]
                    ano_peaks_good = ano_peaks[np.in1d(ano_peaks, anomaly)]
                    ano_peaks_bad = ano_peaks[~np.in1d(ano_peaks, anomaly)]
                    ax.scatter(ano_peaks_good + self.shift, ano_signal[ano_peaks_good], label = "anomaly", c="red", marker='x', zorder = 3)
                    # ax.scatter(ano_peaks_bad, ano_signal[ano_peaks_bad], label = "peak", c="blue", marker='x', zorder = 3)

            ax.legend()

        plt.tight_layout()
        plt.show()

    def compare_ui(self):
        dataset_dropdown = widgets.Dropdown(
            options=self.datafiles,
            description='Dataset:'
        )
        label_dropdown = widgets.Dropdown(
            options=self.labels,
            description='Label:'
        )
        start_text = widgets.BoundedIntText(
            min=self.shift,
            max=100000000,
            value=0,
            description='Start:'
        )
        duration_text = widgets.IntText(
            value=-1,
            description='Duration:'
        )
        # avg_alpha_text = widgets.FloatText(
        #     value=1,
        #     description='Exponential average coef:'
        # )
        # anomaly_dropdown = widgets.Dropdown(
        #     options=["Peaks", "Votes", "Gaussian"],
        #     description="Anomaly method:"
        # )
        run_button = widgets.Button(
            description='Update',
            tooltip='Update'
        )

        out = widgets.Output(layout={'border': '1px solid black'})

        def show_ui():
            with out:
                display(dataset_dropdown,
                        label_dropdown,
                        start_text,
                        duration_text,
                        run_button)

        def run_button_action(b):
            out.clear_output()
            show_ui()

            start = start_text.value - self.shift
            end = start_text.value + duration_text.value - self.shift
            feat = self.labels.index(label_dropdown.value)
            dset = dataset_dropdown.value
            # avg_alpha = avg_alpha_text.value
            # ano_method = anomaly_dropdown.value

            signame = self.labels[feat]
            pred = self.preds[dset]
            targ = self.targs[dset]
            error_nabs = self.errors[dset]
            error = {model: np.absolute(error_nabs[model]) for model in error_nabs}
            # test_sets = get_test_sets(targs) if "test/" not in dset else []

            pred_norm,targ_norm, error_norm = {}, None, {}
            # error_norm = {i : error[i] / np.mean(np.absolute(targ), axis=0) for i in error}

            for model in pred:
                pred_norm[model], targ_norm = normalize_data(pred[model], targ)
                error_norm[model] = normalize_error(error[model])

            pred_len = pred[self.models[0]].shape[0]

            if end > pred_len or duration_text.value <= 0:
                end = pred_len

            # save_all_graphs("/opt/platform/data/jolibrain/data_set_1/graphs/nbeats/", models, pred, targ, error, labels, shift, test_sets)

            with out:
                print("n features: %d, n signals: %d" % (targ.shape[1], pred_len))
                print("selected feature: %d" % feat)
                for model in pred:
                    if len(pred_norm[model]) == len(targ_norm):
                        print("mean error for %s: %f" % (model, get_signal_mean_error(pred_norm[model], targ_norm, feat)))

                if False: # avg_alpha < 1:
                    print("averaging with factor %f" % avg_alpha)
                    error_norm = {i : exp_mean_avg(error_norm[i], avg_alpha) for i in error_norm}

                # Anomalies detection
                # anomalies = {i: compute_anomalies(error_norm[i], error_nabs[i], ano_method) for i in error_norm}

                title = "%s signal from %d to %d " % (signame, start + self.shift, end + self.shift)
                self.display_compare(pred, targ, error, [feat], start, end, title)

        run_button.on_click(run_button_action)
        show_ui()
        return out

    def save_all_graphs(self, dest, pred, targ, error, labels, test_sets):
        """
        Save all signals prediction graph
        """
        max_i =  targ.shape[0] - 1 - shift

        for feat in range(targ.shape[1]):
            # compute mean error
            pred_norm, targ_norm = normalize_data(pred[models[0]][:,feat], targ[:,feat])
            mean_error = np.mean(np.absolute(targ_norm - pred_norm))

            signame = self.labels[feat] + "_" + str(mean_error)
            print(signame)
            self.display_compare(pred, targ, error, [feat], 0, max_i, signame + " whole signal", tests = test_sets, save = dest + signame + ".png")
