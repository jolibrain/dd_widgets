import os
import sys
import csv
import io
import re
import time
import numpy as np
import tqdm
from typing import List
from tqdm import notebook

from . import *

class NBEATS(Timeseries):

    def __init__(self,
                sname: str,
                *,  # unnamed parameters are forbidden
                path: str = "",
                description: str = "NBEATS model",
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
                offset: int = 1,
                test_initialization: bool = False,
                batch_size: int = 50,
                iter_size: int = 1,
                test_batch_size: int = 100,
                loss: str = "L1",
                backcast_timesteps : int = 100,
                forecast_timesteps : int = 100,
                template_params : Dict[str, Any] = {
                    "stackdef" : ["g512", "g512", "b5", "h512"],
                    "backcast_loss_coeff" : 1.0,
                },
                ### Predict parameters
                # predict every x timesteps
                pred_interval : int = 1,
                # predict from past output
                autoregressive : bool = False,
                **kwargs):

        super().__init__(sname, local_vars=locals(), **kwargs)

        # During predict, make prediction every X timesteps
        self.pred_interval = pred_interval

        # set shift (see Timeseries)
        self.shift = backcast_timesteps

    def _create_parameters_input(self):
        return {
            'connector':'csvts',
            'db':False,
            'separator':self.csv_separator.value,
            'forecast_timesteps':self.forecast_timesteps.value,
            'backcast_timesteps':self.backcast_timesteps.value,
            "ignore": eval(self.ignore_columns.value),
        }

    def _create_parameters_mllib(self):
        dic = dict(
            template="nbeats",
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
        dic["template_params"] = eval(self.template_params.value)
        return dic

    def _train_parameters_input(self):
        return {
            "shuffle": True,
            "separator": self.csv_separator.value,
            "db": False,
            "scale": True,
            "offset": self.offset.value,
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
            'forecast_timesteps':self.forecast_timesteps.value,
            'backcast_timesteps':self.backcast_timesteps.value,
        }

    def _predict_parameters_mllib(self):
        return {'net':{'test_batch_size':1}}

    def _predict_parameters_output(self):
        return {}

    def get_preds_targets_nbeats(self, datafile):
        backcast = self.backcast_timesteps.value
        forecast = self.forecast_timesteps.value
        timesteps = backcast + forecast

        with open(datafile) as linefile:
            raw_lines = linefile.readlines()[1:]
            num_lines = len(raw_lines)

        with open(datafile) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)

            raw_data = []
            for data in csv_reader:
                raw_data.append(data)
            raw_data = np.array(raw_data, dtype=np.double)

        col_labels = []
        for l in header:
            if l not in eval(self.ignore_columns.value):
                col_labels.append(get_col(header,l))

        npoints = num_lines-backcast
        nlabels = len(col_labels)

        csvheader=io.StringIO()
        headerwriter = csv.writer(csvheader)
        headerwriter.writerow(header)

        predictions = np.full((npoints,nlabels),0, dtype=np.double)
        predictions_list = []
        for i in range(0,npoints):
            predictions_list.append([])
        npreds =  np.full(npoints,0,dtype=np.double)
        targets = np.full((npoints,nlabels),0, dtype=np.double)

        for nline in self.logger_params.progress_bar(range(0,num_lines-timesteps + self.pred_interval, self.pred_interval)): # sliding window
            # take last iteration into account
            it_points = min(forecast, num_lines - backcast - nline)

            if it_points <= 0:
                continue
            # input
            # csvdata = io.StringIO()
            # datawriter = csv.writer(csvdata)
            # for j in range(0,backcast):
            #    datapoint = raw_data[nline+j]
            #    datawriter.writerow(datapoint)
            data = "".join(raw_lines[nline:nline + backcast])

            # output => forecast
            targets[nline:nline + it_points] = raw_data[nline+backcast:nline + backcast + it_points,col_labels]

            out = self.predict([csvheader.getvalue(), data], enable_logging = False)
            out = self.get_dd_predictions(out)

            pred = np.array([out[0]['series'][i]['out'] for i in range(it_points)], dtype=np.double)
            predictions[nline:nline + forecast] += pred
            npreds[nline:nline+forecast] += 1

            predictions[nline:nline + it_points] += pred
            npreds[nline:nline+it_points] += 1

            for j in range(0,it_points):
                #print("update pred " + str(nline+j))
                for k in range(0,len(pred[0])):
                    # predictions[nline+j,k] =  predictions[nline+j,k]  + pred[j][k]
                    predictions_list[nline+j].append(pred[j][k])

        npreds[npreds == 0] = 1
        for k in range(0, len(col_labels)):
            predictions[:,k] = predictions[:,k] / npreds

        return predictions, predictions_list, npreds, targets

    def get_nbeats_autoregression(self, datafile, start, stop):
        """
        get_preds_targets but with autoregression
        start = timestep where we start predicting
        stop = timestep where we stop predicting
        """
        backcast = self.backcast_timesteps.value
        forecast = self.forecast_timesteps.value
        timesteps = backcast + forecast

        with open(datafile) as linefile:
            raw_lines = linefile.readlines()[1:]
            num_lines = len(raw_lines)

        if start < backcast:
            raise ValueError("start %d < backcast %d" % (start, backcast))

        if stop < 0:
            stop = num_lines

        with open(datafile) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)

            raw_data = []
            for data in csv_reader:
                raw_data.append(data)
            raw_data = np.array(raw_data, dtype=np.double)

        col_labels = []
        for l in header:
            if l not in eval(self.ignore_columns.value):
                col_labels.append(get_col(header,l))

        npoints = stop - start
        ntarg = min(npoints, num_lines - backcast)
        nlabels = len(col_labels)

        csvheader=io.StringIO()
        headerwriter = csv.writer(csvheader)
        headerwriter.writerow(header)

        predictions = np.full((npoints,nlabels),0, dtype=np.double)
        targets = np.full((ntarg,nlabels),0, dtype=np.double)

        prev = raw_data[start-backcast:start,:]

        for nline in self.logger_params.progress_bar(range(start, stop, forecast)):
            it_points = min(forecast, stop - nline)
            pred_line = nline - start

            # input
            csvdata = io.StringIO()
            datawriter = csv.writer(csvdata)
            for j in range(0,backcast):
                datapoint = prev[- backcast + j]
                datawriter.writerow(datapoint)

            # output => forecast
            targ_points = min(forecast, num_lines - nline)
            if nline + targ_points <= num_lines:
                targets[pred_line:pred_line + targ_points] = raw_data[nline:nline + targ_points,col_labels]

            out = self.predict([csvheader.getvalue(),csvdata.getvalue()])
            out = self.get_dd_predictions(out)

            pred = np.array([out[0]['series'][i]['out'] for i in range(it_points)], dtype=np.double)
            predictions[pred_line:pred_line + it_points] = pred

            # Update data signal with prediction (autoregressive)
            if it_points == forecast: # no need to update for the last iteration
                prev = np.concatenate(
                        (prev[- backcast + forecast:],
                        np.full((forecast,prev.shape[1]), 0, dtype=np.double))
                    )
                prev[-forecast:,col_labels] = predictions[pred_line: pred_line + forecast]

        return predictions, targets

    def predict_file(self, datapath):
        if self.autoregressive.value:
            pred, targ = self.get_nbeats_autoregression(datapath, self.backcast_timesteps.value, -1)
        else:
            pred, predictions_list, npreds, targ = self.get_preds_targets_nbeats(datapath)
        return pred, targ
