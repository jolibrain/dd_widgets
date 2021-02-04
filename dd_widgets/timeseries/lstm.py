import os
import sys
import csv
import io
import re
import time
import numpy as np

from . import *

class LSTM(Timeseries):

    def __init__(self,
                pred_distance,
                layers = ["L256", "L256"],
                timesteps = 400,
                **kwargs):

        super().__init__(**kwargs)
        self.layers = layers
        # distance of prediction in timesteps
        self.pred_distance = pred_distance
        # length of sequences during learning
        self.timesteps = timesteps

        # set shift (see Timeseries)
        self.shift = pred_distance

    def create_service(self, model_name, predict = False):
        sname = self.get_predict_sname() if predict else self.sname
        parameters_input = {
            'connector':'csvts',
            'timesteps':self.timesteps,
            'db':False,
            'separator':',',
            'ignore':self.ignored_cols,
            'label':self.target_cols
        }
        parameters_mllib = {
            'loss':'L1',
            'template':'recurrent',
            'db':False,
            'gpuid':self.gpuid,
            'gpu':True,
            'layers': self.layers,
            'dropout':0.0,
            'regression':True
        }
        parameters_output = {'store_config': not predict}
        model = {
            'repository':os.path.join(self.models_dir, model_name),
            'create_repository': True
        }
        try:
            creat = self.dd.put_service(sname,model,'airbus timeserie prediction','torch',parameters_input,parameters_mllib,parameters_output)
            self.log_progress(creat)
        except Exception as e:
            raise e

    def train(self, data):
        parameters_input = {
            'shuffle':True,
            'separator':',',
            'scale':True,
            'offset':self.offset,
            'db':False,
            'separator':',',
            'ignore':self.ignored_cols,
            'label':self.target_cols
        }
        solver_params = {
            'test_interval':2000,
            'snapshot':2000,
            "iterations":500000,
            'base_lr':0.001,
            'solver_type':'RANGER_PLUS',
            'clip': True,
            'test_initialization':False
        }
        solver_params.update(self.solver_params)
        parameters_mllib = {
            'gpu':True,
            'gpuid':self.gpuid,
            'net': {'batch_size':self.batch_size,'test_batch_size':min(self.batch_size, 30) },
            'solver': solver_params
        }
        parameters_output = {
            'measure':['L1','L2']
        }
        self.dd.post_train(self.sname, data, parameters_input, parameters_mllib, parameters_output, jasync=True)

    def get_timeserie_results_lstm(self, data, continuation = True):
        parameters_input = {
            'connector':'csvts',
            'separator':',',
            'scale':True,
            'db':False,
            'timesteps':1,
            'continuation':continuation
        }
        parameters_mllib = {'net':{'test_batch_size':1},'cudnn':True}
        parameters_output = {}
        res = self.dd.post_predict(self.get_predict_sname(),data,parameters_input,parameters_mllib,parameters_output)
        if res["status"]["code"] != 200:
            print(res)
        return res

    def get_pred_targets_lstm(self, datafile, nsteps_before_predict, npredictions):
        csvfile = open(datafile)
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)
        col_labels = []
        for l in self.target_cols:
            col_labels.append(get_col(header,l))

        nsteps = 0
        csvheader=io.StringIO()
        csvdata= io.StringIO()

        headerwriter = csv.writer(csvheader)
        datawriter=csv.writer(csvdata)
        headerwriter.writerow(header)
        predictions = []
        targets = []
        cont = False
        npr = 0

        for data in csv_reader:
            targets.append([float(data[i]) for i in col_labels])
            npr = npr + 1
            datawriter.writerow(data)
            nsteps = nsteps + 1

            if nsteps == nsteps_before_predict:
                out = self.get_timeserie_results_lstm([csvheader.getvalue(),csvdata.getvalue()], cont)
                out = self.get_dd_predictions(out)
                cont = True
                predictions.extend([out[0]['series'][i]['out'] for i in range(nsteps_before_predict)])
                nsteps = 0
                if npr >= npredictions:
                    break
                csvdata = io.StringIO()
                datawriter = csv.writer(csvdata)

        if nsteps != 0:
            out = self.get_timeserie_results_lstm([csvheader.getvalue(),csvdata.getvalue()], cont)
            out = self.get_dd_predictions(out)
            predictions.extend([out[0]['series'][i]['out'] for i in range(len(out[0]['series']))])

        return np.asarray(predictions), np.asarray(targets)

    # TODO Merge common parts with nbeats into timeseries?
    def predict_all(self, nsteps_before_predict = 10000, override = False):
        if not override:
            self.load_targets()
            self.load_preds_errors()

        predicted_models = []
        self.delete_service(predict = True)

        for model in self.progress_bar(self.models):
            self.create_service(model, predict = True)

            for datafile in self.progress_bar(self.datafiles):
                if datafile not in self.preds:
                    self.preds[datafile] = {}
                    self.errors[datafile] = {}

                if model in self.preds[datafile] and not override:
                    self.log_progress("skipping predict for %s with model %s: already exist" % (datafile, model))
                    continue

                datapath = os.path.join(self.datadir, datafile)
                pred, targ = self.get_pred_targets_lstm(datapath, nsteps_before_predict, float("inf"))
                self.preds[datafile][model] = pred
                self.errors[datafile][model] = pred - targ
                self.targs[datafile] = targ

            predicted_models.append(model)
            self.delete_service(predict = True)

        self.dump_model_preds(predicted_models)
        self.log_job_done()

    ## Autoencoder part (TODO or remove)

    def extract_layer_timeseries(dd, sname, data, extract_layer, continuation = True):
        parameters_input = {'connector':'csvts','separator':',','scale':True,'db':False, 'timesteps':1, 'continuation':continuation}
        parameters_mllib = {'net':{'test_batch_size':1}, "extract_layer": extract_layer,'cudnn':True}
        parameters_output = {}
        res = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
        if res["status"]["code"] == 500:
            print(res)
        return res

    # DEFAULT_EXTRACT_LAYER = "LSTM2_final_h"

    def get_embeddings(dd, sname, datafile, timesteps, labels, extract_layer, npredictions = float("inf")):
        csvfile = open(datafile)
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)
        col_labels = []
        for l in labels:
            col_labels.append(get_col(header,l))

        nsteps = 0
        csvheader=io.StringIO()
        csvdata= io.StringIO()

        headerwriter = csv.writer(csvheader)
        datawriter=csv.writer(csvdata)
        headerwriter.writerow(header)
        embeddings = []
        targets = []
        cont = False
        npr = 0

        for data in self.progress_bar(csv_reader):
            targets.append([float(data[i]) for i in col_labels])
            npr = npr + 1
            datawriter.writerow(data)
            nsteps = nsteps + 1

            if nsteps == timesteps:
                out = extract_layer_timeseries(dd,sname, [csvheader.getvalue(),csvdata.getvalue()], extract_layer, cont)
                out = self.get_dd_predictions(out)
                embeddings.append(out[0]["vals"])
                # predictions.extend([out[0]['series'][i]['out'] for i in range(nsteps_before_predict)])
                cont = True
                nsteps = 0

                if npr >= npredictions:
                    break

                csvdata = io.StringIO()
                datawriter = csv.writer(csvdata)

        out = extract_layer_timeseries(dd,sname, [csvheader.getvalue(),csvdata.getvalue()], extract_layer, cont)
        out = self.get_dd_predictions(out)

        embeddings.append(out[0]["vals"])
        # predictions.extend([out[0]['series'][i]['out'] for i in range(len(out[0]['series']))])

        return embeddings, np.asarray(targets)

    def one_class_svm_on_embeddings(embeddings, **kwargs):
        ocsvm = OneClassSVM(**kwargs)
        ocsvm.fit(embeddings)
        return ocsvm, ocsvm.predict(embeddings)

    def plot_autoencoder_anomalies(target, class_ids, timesteps, tstart, tend, signal, title, test_sets = []):
        models = [""]
        fig = plt.figure(figsize = (len(models) * 6, 4))
        axs = fig.subplots(1, len(models))
        # fig.title(title)
        fig.set_facecolor("w")

        for i in range(len(models)):
            model = models[i]

            ax = axs[i] if len(models) > 1 else axs

            ax.set_title(title + "\n" + model)
            ax.set_xlabel("time")
            ax.set_ylabel("amplitude")
            ax.plot(target[tstart:tend,signal], label="target")

            # plot anomaly
            # ax.scatter(anomaly, error[anomaly,signal], label = "anomalies", c="red", marker='x')
            for i in range(len(class_ids)):
                ano_start = i * timesteps
                ano_end = (i + 1) * timesteps

                if class_ids[i] == -1 and (tstart < ano_start < tend or tstart < ano_end < tend):
                    ax.axvspan(ano_start - tstart, ano_end - tstart, facecolor='red', alpha=0.3)

            for test_start, test_end in test_sets:
                if tstart < test_start < tend or tstart < test_end < tend:
                    ax.axvspan(test_start - tstart, test_end - tstart, facecolor='green', alpha=0.3)

            ax.legend()

        plt.tight_layout()
        plt.show()
