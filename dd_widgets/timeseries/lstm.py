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
                sname: str,
                *,  # unnamed parameters are forbidden
                path: str = "",
                description: str = "Recurrent model",
                model_repo: Path = None,
                mllib: str = "torch",
                training_repo: Path = None,
                testing_repo: List[Path] = None,
                host: str = "localhost",
                port: int = 1234,
                gpuid: GPUIndex = 0,
                nclasses: int = -1,
                label_columns: List[str] = [],
                ignore_columns: List[str] = [],
                layers : List[str] = ["L256", "L256"],
                csv_separator: str = ",",
                solver_type: Solver = "RANGER_PLUS",
                sam : bool = False,
                lookahead : bool = True,
                lookahead_steps : int = 6,
                lookahead_alpha : float = 0.5,
                rectified : bool = True,
                decoupled_wd_periods : int = 4,
                decoupled_wd_mult : float = 2.0,
                lr_dropout : float = 1.0,
                resume: bool = False,
                base_lr: float = 1e-3,
                warmup_lr: float = 1e-5,
                warmup_iter: int = 0,
                iterations: int = 500000,
                snapshot_interval: int = 5000,
                test_interval: int = 5000,
                timesteps: int = 400,
                offset: int = 1,
                test_initialization: bool = False,
                batch_size: int = 50,
                iter_size: int = 1,
                test_batch_size: int = 100,
                loss: str = "L1",
                ### Predict parameters
                # distance between prediction and target
                pred_distance : int = 100,
                **kwargs):

        super().__init__(sname, local_vars=locals(), **kwargs)
        # distance of prediction in timesteps (used only for display)
        self.pred_distance = pred_distance

        # set shift (see Timeseries)
        self.shift = pred_distance

    def _create_parameters_input(self):
        return {
            "connector": "csvts",
            "db": False,
            "label": eval(self.label_columns.value),
            "ignore": eval(self.ignore_columns.value),
            "timesteps": self.timesteps.value,
        }

    def _create_parameters_mllib(self):
        dic = dict(
            template="recurrent",
            db=False,
            dropout=0.0,
            loss=self.loss.value,
        )
        dic["gpu"] = True
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic["gpuid"] = (
            list(self.gpuid.index)
            if len(self.gpuid.index) > 1
            else self.gpuid.index[0]
        )
        dic["timesteps"] = self.timesteps.value
        dic["layers"] = eval(self.layers.value)  #'["L50", "L50", "A3", "L3"]'
        return dic

    def _train_parameters_input(self):
        return {
            "shuffle": True,
            "separator": self.csv_separator.value,
            "db": False,
            "scale": True,
            "offset": self.offset.value,
            "timesteps": self.timesteps.value,
        }

    def _train_parameters_mllib(self):
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic = {
            "gpu": True,
            "gpuid": (
                list(self.gpuid.index)
                if len(self.gpuid.index) > 1
                else self.gpuid.index[0]
            ),
            "resume": self.resume.value,
            "timesteps": self.timesteps.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "sam" : self.sam.value,
                "test_initialization": self.test_initialization.value,
            },
        }

        return dic

    def _train_parameters_output(self):
        return {"measure": ["L1"]}

    def _predict_parameters_input(self):
        return {
            'connector':'csvts',
            'separator':self.csv_separator.value,
            'scale':True,
            'db':False,
            'continuation': self.continuation
        }

    def _predict_parameters_mllib(self):
        return {
            'net':{'test_batch_size':1},
            'cudnn':True
        }

    def _predict_parameters_output(self):
        return {}

    def get_pred_targets_lstm(self, datafile, nsteps_before_predict, npredictions):
        csvfile = open(datafile)
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)
        col_labels = []
        for l in eval(self.label_columns.value):
            col_labels.append(get_col(header,l))

        nsteps = 0
        csvheader=io.StringIO()
        csvdata= io.StringIO()

        headerwriter = csv.writer(csvheader)
        datawriter=csv.writer(csvdata)
        headerwriter.writerow(header)
        predictions = []
        targets = []
        self.continuation = False
        npr = 0

        for data in csv_reader:
            targets.append([float(data[i]) for i in col_labels])
            npr = npr + 1
            datawriter.writerow(data)
            nsteps = nsteps + 1

            if nsteps == nsteps_before_predict:
                out = self.predict([csvheader.getvalue(),csvdata.getvalue()], enable_logging = False)
                out = self.get_dd_predictions(out)
                self.continuation = True
                predictions.extend([out[0]['series'][i]['out'] for i in range(nsteps_before_predict)])
                nsteps = 0
                if npr >= npredictions:
                    break
                csvdata = io.StringIO()
                datawriter = csv.writer(csvdata)

        if nsteps != 0:
            out = self.predict([csvheader.getvalue(),csvdata.getvalue()])
            out = self.get_dd_predictions(out)
            predictions.extend([out[0]['series'][i]['out'] for i in range(len(out[0]['series']))])

        return np.asarray(predictions), np.asarray(targets)

    def predict_file(self, datapath, nsteps_before_predict = 10000):
        pred, targ = self.get_pred_targets_lstm(datapath, nsteps_before_predict, float("inf"))
        return pred, targ



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
