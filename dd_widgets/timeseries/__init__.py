import os
import sys
import csv
import io
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import tqdm
from tqdm import notebook
import IPython.display as jp_display
import ipywidgets as widgets

from . import anomalies as ano

from dd_client import DD

# Helper functions

def delete_service(dd, sname):
    try:
        print(dd.delete_service(sname))
        return True
    except Exception as e:
        if e.response.status_code != 404:
            raise
        return False

# To dd_client?
def wait_training_over(dd, sname, sleeptime=5):
    while True:
        try:
            res = dd.get_train(sname)
            if res["head"]["status"] != "running":
                return
        except HTTPError as e:
            print(e)
            if e.response.status_code != 404:
                raise
        # print("Waiting end of training of %s" % sname)
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

def load_target(datafile,labels):
    csvfile = open(datafile)
    csv_reader = csv.reader(csvfile, delimiter=',')
    header = next(csv_reader)
    col_labels = []
    targets = []
    for l in labels:
        col_labels.append(get_col(header,l))

    for data in csv_reader:
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

def concat_all_datafiles(errors, datafiles):
    """
    Create one error array by juxtaposition of all errors on
    the train set. This array may be used by the ErrorNormalizationModel
    to compute the coefficients.

    errors[datafile][model] -> concatened_error[model]
    """
    result = {}

    for datafile in datafiles:
        error = errors[datafile]

        for model in error:
            if model not in result:
                result[model] = error[model]
            else:
                result[model] = np.concatenate((result[model], error[model]), axis = 0)

    return result

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


class AnomalyParameters:
    def __init__(self,
                method = "threshold_norm",
                smooth_factor = 20,
                threshold = 3,
                n_anomalies = 40,
                ignore = []):

        self.method = method
        self.smooth_factor = smooth_factor
        self.labels = []
        self.ignore = ignore

        # Gauss parameters
        self.threshold = threshold

        # Votes & peaks parameters
        # Total number of anomalies detected
        self.n_anomalies = n_anomalies
        self.err_norm_model = None

    def available_methods():
        return ["threshold_norm", "threshold", "peaks", "votes", "gaussian"]

    def fit(self, error):
        """
        Fit error model on given error
        This consists in computing mean and standard deviation for every
        signal.
        """
        err_norm_model = ano.ErrorNormalizationModel()
        c = self.smooth_factor + 1
        conv = [1 / c] * c

        col_ids = []
        for lbl in self.labels:
            if lbl not in self.ignore:
                col_ids.append(get_col(self.labels,lbl))

        error = error[:,col_ids]
        err_norm_model.fit(error, conv, display = True)

        self.err_norm_model = err_norm_model

    def compute_anomalies(self, error, display = False):
        """
        Compute anomalies of signal based on the prediction
        error on the signal
        error: the error of signal
        returns: list of anomaly dates (in timesteps), signal allowing to
            compute anomalies (typically avg mean error), list of other
            potential anomaly dates
        """
        c = self.smooth_factor + 1
        conv = [1 / c] * c
        ano_signal = None
        ano_peaks = None

        # eliminate ignored labels
        col_ids = []
        for lbl in self.labels:
            if lbl not in self.ignore:
                col_ids.append(get_col(self.labels,lbl))

        error = error[:,col_ids]
        error_norm = normalize_error(np.absolute(error))

        if self.method == "threshold_norm":
            err_norm_model = self.err_norm_model

            if not err_norm_model:
                err_norm_model = ano.ErrorNormalizationModel()
                err_norm_model.fit(error, conv, display = display)

            anomalies, ano_signal, ano_peaks = err_norm_model.anomaly_dates(error, self.threshold, conv, display = display)
        elif self.method == "gaussian":
            gauss_model = self.err_norm_model

            if not gauss_model:
                gauss_model = ano.ErrorNormalizationModel()
                gauss_model.fit(error, conv, fit_gauss = True, display = display)

            anomalies, ano_signal, ano_peaks = gauss_model.anomaly_dates(error, self.threshold, conv, display = display)
        elif self.method == "peaks":
            # ano_signal = error_peaks(error_norm, conv=conv)
            # anomalies = anomaly_dates(ano_signal, 20)
            anomalies, ano_signal, ano_peaks = ano.anomaly_dates_peaks(error_norm, self.n_anomalies,conv=conv)
        elif self.method == "votes":
            ano_signal = ano.error_vote(error_norm, self.n_anomalies, conv=conv)
            anomalies = ano.anomaly_dates_votes(ano_signal, self.n_anomalies)
        elif self.method == "threshold":
            error_norm = ano.sum_conv_error(error_norm, conv) / len(col_ids)
            ano_signal = error_norm
            anomalies = np.where(error_norm > self.threshold)[0]
        else:
            raise ValueError(
                "Unknown anomaly method: %s. Available methods are %s"
                % (self.method, str(AnomalyParameters.available_methods()))
            )

        return anomalies, ano_signal, ano_peaks

    def get_anomaly_results(self, pred_anom, targ_anom):
        """
        Compute accuracy & recall of the model in predicting anomalies.
        pred_anom: list of anomalies as obtained by the compute_anomalies()
            method
        targ_anom: bounds of anomalies to detect (ground truth)
        Return: true positive / false negative / false positive
        """
        found = [False] * len(targ_anom)
        fps = []

        for anom in pred_anom:
            match = False

            for i in range(len(targ_anom)):
                start, end = targ_anom[i]

                if start <= anom <= end:
                    found[i] = True
                    match = True
                    break

            if not match:
                fps.append(anom)

        tp = sum([1 for i in found if i])
        fn = len(targ_anom) - tp
        return tp, fn, len(fps)

    def get_anomaly_score(self, pred_anom, targ_anom):
        """
        Compute accuracy & recall of the model in predicting anomalies.
        pred_anom: list of anomalies as obtained by the compute_anomalies()
            method
        targ_anom: bounds of anomalies to detect (ground truth)
        Return: accuracy, recall
        """
        tp, fn, fp = self.get_anomaly_results(pred_anom, targ_anom)
        return tp / (tp + fn), tp / (tp + fp)

class Timeseries:
    """
    Train and predict timeseries in jupyter.
    """
    def __init__(self,
                sname,
                host = "127.0.0.1",
                port = "8080",
                proto = 0,
                api_path = "",
                model = None,
                models = [],
                models_dir = "/opt/platform/models/",
                datafiles = [],
                datadir = "",
                output_dir = "/temp/predictions/",
                columns = [],
                target_cols = [],
                ignored_cols = [],
                offset=50,
                gpuid = 0,
                autoregressive = False,
                sam = False,
                batch_size = 50,
                iter_size = 1,
                iterations = 500000,
                base_lr=0.001,
                test_interval = 5000,
                anomaly_params = AnomalyParameters(),
                display_progress = True):

        self.sname = sname
        self.models = models
        self.models_dir = models_dir
        self.datafiles = datafiles
        self.datadir = datadir
        self.output_dir = output_dir
        self.columns = columns
        self.target_cols = target_cols
        self.ignored_cols = ignored_cols
        self.offset = offset
        self.batch_size = batch_size
        self.gpuid = gpuid
        self.autoregressive = autoregressive
        self.display_progress = display_progress

        self.solver_params = {
            "iter_size": iter_size,
            "iterations": iterations,
            "base_lr": base_lr,
            "test_interval": test_interval,
            "sam": sam
        }

        """
        shift: How much the target is being shifted. As models predict
        at different horizons, shift enable to compare the same sections
        of the targets even if the target is shifted by a certain number
        of timesteps.
        """
        self.shift = 0

        # error based anomaly detection
        self.anomaly_params = anomaly_params
        self.anomaly_params.labels = self.target_cols

        # dict {dataset: target}
        self.targs = {}
        # dict { dataset: {model: preds / errors}}
        self.preds = {}
        # signed error
        self.errors = {}

        self.dd = DD(host, port, proto, api_path)

    def log_progress(self, msg):
        if self.display_progress:
            print(msg)

    def log_job_done(self):
        # jp_display.clear_output(wait = True)
        self.log_progress("Done!")

    def progress_bar(self, generator, leave = False):
        if self.display_progress:
            return tqdm.notebook.tqdm(generator, leave = leave)
        else:
            return generator

    def get_predict_sname(self):
        return self.sname + "_predict"

    def delete_service(self, predict = False):
        sname = self.get_predict_sname() if predict else self.sname
        if not delete_service(self.dd, sname):
            self.log_progress("Service %s does not exist" % sname)

    def get_dd_predictions(self, dd_pred_response):
        """
        Get predictions from a json response, report error if any.
        """
        if "body" not in dd_pred_response: # error
            raise RuntimeError(dd_pred_response)
        return dd_pred_response['body']['predictions']

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
        for model in self.progress_bar(models):
            model_out_dir = os.path.join(self.output_dir, model)

            # print("creating", model_out_dir, "...")
            os.makedirs(model_out_dir, exist_ok = True)

            for datafile in self.progress_bar(self.datafiles):
                pred_out_file, err_out_file = self.get_dump_filenames(datafile, model)

                # needed if "test/"
                try:
                    os.mkdir(os.path.dirname(pred_out_file))
                except FileExistsError:
                    pass

                dump_data(self.preds[datafile][model], self.target_cols, pred_out_file)
                dump_data(self.errors[datafile][model], self.target_cols, err_out_file)


    def load_targets(self, columns = None):
        if columns is None:
            columns = self.target_cols;

        # return dict for each datafile with target
        for datafile in self.progress_bar(self.datafiles):
            targ_file = os.path.join(self.datadir, datafile)
            if not os.path.exists(targ_file):
                self.log_progress("cannot load target file %s: does not exist" % datafile)
                continue

            self.targs[datafile] = np.array(load_target(targ_file, columns))

    def load_preds_errors(self):
        # return dict of dict for each datafile for each model
        for model in self.models:
            model_out_dir = os.path.join(self.output_dir, model)

            if not os.path.exists(model_out_dir):
                self.log_progress("cannot load predictions for %s: model directory not found" % model)
                continue

            for datafile in self.progress_bar(self.datafiles):
                if datafile not in self.preds:
                    self.preds[datafile] = {}
                    self.errors[datafile] = {}

                pred_out_file, err_out_file = self.get_dump_filenames(datafile, model)

                if not os.path.exists(pred_out_file):
                    self.log_progress("cannot load predictions file %s" % pred_out_file)
                    continue

                # FIXME pass columns to load_data (in case it changed before)
                self.preds[datafile][model] = load_data(pred_out_file)
                self.errors[datafile][model] = load_data(err_out_file)

    def reset_pred_targ_error(self):
        """
        Remove all precomputed target / pred / error.
        """
        self.preds = {}
        self.targs = {}
        self.errors = {}

    def learn_anomalies(self, model, datafiles):
        """
        Precompute thresholds for anomaly detection on given files.
        """
        # TODO learn all models?
        # TODO allow datafiles from everywhere
        error = concat_all_datafiles(self.errors, datafiles)
        self.anomaly_params.fit(error[model])

    # Plot / Widgets

    # TODO Add shift option
    def plot_dataset(self, targ, signals, tstart = None, tend = None, title = None, save = None):
        if not tstart:
            tstart = 0
        if not tend or tend > len(targ):
            tend = len(targ)
        indices = np.arange(tstart, tend)
        if title != None:
            plt.title(title)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        for s in signals:
            plt.plot(indices, targ[tstart:tend,s], label="signal")
        plt.legend()
        plt.gcf().set_facecolor("w")

        if save:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()

    def dataset_ui(self):
        if len(self.targs) == 0:
            self.load_targets()

        dataset_dropdown = widgets.Dropdown(
            options=self.datafiles,
            description='Dataset:'
        )
        label_dropdown = widgets.Dropdown(
            options=self.target_cols,
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
            feat = self.target_cols.index(label_dropdown.value)
            dset = dataset_dropdown.value

            signame = self.target_cols[feat]
            targ = self.targs[dset]

            if end >= len(targ) or duration_text.value <= 0:
                end = len(targ)

            with out:
                title = "signal %s from %d to %d" % (signame, start, end)
                self.plot_dataset(targ, [feat], start, end, title)

        run_button.on_click(run_button_action)
        show_ui()
        return out

    def save_dataset_graphs(self, location, tstart = None, tend = None):
        if len(self.targs) == 0:
            self.load_targets()

        dsetid = 0

        for dset in self.progress_bar(self.datafiles):
            for label in self.target_cols:
                feat = self.target_cols.index(label)
                title = "%s - %s" % (dset, label)

                if tstart:
                    title += " from %d to %d" % (tstart, tend)
                self.plot_dataset(self.targs[dset], [feat], tstart, tend, title,
                    os.path.join(location, "graph_%d_%s.png" % (dsetid, label)))

            dsetid += 1

    def plot_predictions(self,
                    preds,
                    targ,
                    errors,
                    signals,
                    tstart,
                    tend,
                    title,
                    anomalies = None,
                    targ_anom = None,
                    tests = None,
                    save = False):

        """
        preds: dict {model: preds}
        targ: raw target
        signal: id of the signal (= line to display)
        shift: start value offset for x axis
        this method is used to display one specific dataset
        """
        # plt.subplots_adjust(hspace=1)
        indices = np.arange(self.shift + tstart, self.shift + tend)

        nh = 2
        fig = plt.figure(figsize = (len(self.models) * 9, nh * 6))
        axs = fig.subplots(nh, len(self.models))

        # fig.title(title)
        fig.set_facecolor("w")

        for i in range(len(self.models)):
            model = self.models[i]
            pred = preds[model]

            ax1 = axs[0, i] if len(self.models) > 1 else axs[0]
            ax1.set_title(title + "\n" + model)
            ax1.set_xlabel("time")
            ax1.set_ylabel("amplitude")

            for s in signals:
                s_pred = pred[tstart:tend,s]
                ax1.plot(indices[:len(s_pred)], s_pred, label="pred", zorder=2)
                s_targ = targ[tstart:tend,s]
                ax1.plot(indices[:len(s_targ)], s_targ, label="target", zorder=1)

            ax1.legend()

            ax2 = axs[1, i] if len(self.models) > 1 else axs[1]
            ax2.set_title(title + " error\n" + model)
            ax2.set_xlabel("time")
            ax2.set_ylabel("amplitude")

            if anomalies:
                hwidth = (tend - tstart) / 200
                anomaly, ano_signal, ano_peaks = anomalies[model]

                # TODO both testset and target anomalies are in green
                if targ_anom:
                    for a, b in targ_anom:
                        a = max(tstart + self.shift, a)
                        b = min(tend + self.shift, b)

                        if a < b:
                            ax1.axvspan(a, b, facecolor='green', alpha=0.3)
                            ax2.axvspan(a, b, facecolor='green', alpha=0.3)

                # plot anomaly
                for i in anomaly:
                    if tstart <= i < tend:
                        ax1.axvspan(i - hwidth + self.shift, i + hwidth + self.shift, facecolor='red', alpha=0.3)
                        ax2.axvspan(i - hwidth + self.shift, i + hwidth + self.shift, facecolor='red', alpha=0.3)

                if ano_signal is not None:
                    # ano_signal = ano_signal / max(0.0001, np.max(np.absolute(ano_signal))) * np.max(target)
                    ax2.plot(indices, ano_signal[tstart:tend], label="mean error")

                    if ano_peaks is not None:
                        ano_peaks = ano_peaks[np.logical_and(tstart < ano_peaks, ano_peaks < tend)]
                        ano_peaks_good = ano_peaks[np.in1d(ano_peaks, anomaly)]
                        ano_peaks_bad = ano_peaks[~np.in1d(ano_peaks, anomaly)]
                        ax2.scatter(ano_peaks_good + self.shift, ano_signal[ano_peaks_good], label = "anomaly", c="red", marker='x', zorder = 3)
                        # ax.scatter(ano_peaks_bad, ano_signal[ano_peaks_bad], label = "peak", c="blue", marker='x', zorder = 3)
            else: # anomalies
                error = errors[model]

                for s in signals:
                    s_error = error[tstart:tend,s]
                    ax2.plot(indices[:len(s_error)], s_error, label="error")

            ax2.legend()

            for ax in [ax1, ax2]:
                if tests:
                    for test_start, test_end in tests:
                        test_start = min(max(test_start, tstart), tend)
                        test_end = min(max(test_end, tstart), tend)
                        ax.axvspan(test_start + self.shift, test_end + self.shift, facecolor='green', alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()

    def forecast_ui(self):
        dataset_dropdown = widgets.Dropdown(
            options=self.datafiles,
            description='Dataset:'
        )
        label_dropdown = widgets.Dropdown(
            options=self.target_cols,
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
            out.clear_output()
            with out:
                display(dataset_dropdown,
                        label_dropdown,
                        start_text,
                        duration_text,
                        run_button)

        def run_button_action(b):
            show_ui()

            start = start_text.value - self.shift
            end = start_text.value + duration_text.value - self.shift
            feat = self.target_cols.index(label_dropdown.value)
            dset = dataset_dropdown.value
            # avg_alpha = avg_alpha_text.value
            # ano_method = anomaly_dropdown.value

            signame = self.target_cols[feat]
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

            with out:
                print("n features: %d, n signals: %d" % (targ.shape[1], pred_len))
                print("selected feature: %d" % feat)
                for model in pred:
                    if len(pred_norm[model]) == len(targ_norm):
                        print("mean error for %s: %f" % (model, get_signal_mean_error(pred_norm[model], targ_norm, feat)))

                title = "%s signal from %d to %d " % (signame, start + self.shift, end + self.shift)
                self.plot_predictions(pred, targ, error, [feat], start, end, title)

        run_button.on_click(run_button_action)
        show_ui()
        return out

    def anomalies_ui(self, targ_anom = None):
        dataset_dropdown = widgets.Dropdown(
            options=self.datafiles,
            description='Dataset:'
        )
        label_dropdown = widgets.Dropdown(
            options=self.target_cols,
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
            out.clear_output()
            with out:
                display(dataset_dropdown,
                        label_dropdown,
                        start_text,
                        duration_text,
                        run_button)

        def run_button_action(b):
            show_ui()

            start = start_text.value - self.shift
            end = start_text.value + duration_text.value - self.shift
            feat = self.target_cols.index(label_dropdown.value)
            dset = dataset_dropdown.value

            signame = self.target_cols[feat]
            pred = self.preds[dset]
            targ = self.targs[dset]
            error = self.errors[dset]

            pred_len = pred[self.models[0]].shape[0]

            if end > pred_len or duration_text.value <= 0:
                end = pred_len

            with out:
                print("n features: %d, n signals: %d" % (targ.shape[1], pred_len))
                print("selected feature: %d" % feat)

                # Anomalies detection
                anomalies = {i: self.anomaly_params.compute_anomalies(error[i], display = True) for i in error}

                if targ_anom:
                    for i in anomalies:
                        pred_anom = [ano + self.shift for ano in anomalies[i][0]]
                        print("true pos, false neg, false pos:", self.anomaly_params.get_anomaly_results(pred_anom, targ_anom))

                title = "%s signal from %d to %d " % (signame, start + self.shift, end + self.shift)
                # normalize_error = normalize only one signal. The name can be changed in the future.
                self.plot_predictions(pred, targ, None, [feat], start, end, title, anomalies, targ_anom)

        run_button.on_click(run_button_action)
        show_ui()
        return out

    # TODO make this method more practical to use from outside the class
    def save_all_graphs(self, dest, pred, targ, error, labels, test_sets):
        """
        Save all signals prediction graph
        """
        max_i =  targ.shape[0] - 1 - shift

        for feat in range(targ.shape[1]):
            # compute mean error
            pred_norm, targ_norm = normalize_data(pred[models[0]][:,feat], targ[:,feat])
            mean_error = np.mean(np.absolute(targ_norm - pred_norm))

            signame = self.target_cols[feat] + "_" + str(mean_error)
            print(signame)
            self.display_compare(pred, targ, error, [feat], 0, max_i, signame + " whole signal", tests = test_sets, save = dest + signame + ".png")
