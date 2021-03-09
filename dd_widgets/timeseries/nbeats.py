import os
import sys
import csv
import io
import re
import time
import numpy as np
import tqdm
from tqdm import notebook

from . import *

class NBEATS(Timeseries):

    def __init__(self,
                backcast,
                forecast,
                template_params = ["g512", "g512", "b5", "h512"],
                backcast_loss_coeff = 1.0,
                pred_interval = 20,
                **kwargs):

        super().__init__(**kwargs)
        self.backcast = backcast
        self.forecast = forecast
        self.template_params = template_params
        self.backcast_loss_coeff = backcast_loss_coeff
        # During predict, make prediction every X timesteps
        self.pred_interval = pred_interval

        # set shift (see Timeseries)
        self.shift = backcast

    def create_service(self, model_name, predict = False):
        sname = self.get_predict_sname() if predict else self.sname
        parameters_input = {
            'connector':'csvts',
            'db':False,
            'separator':',',
            'ignore': self.ignored_cols,
            'forecast_timesteps':self.forecast,
            'backcast_timesteps':self.backcast
        }
        parameters_mllib = {
            'loss':'L1',
            'template':'nbeats',
            'template_params':{"stackdef": self.template_params, "backcast_loss_coeff": self.backcast_loss_coeff},
            'db':False,
            'gpuid':self.gpuid,
            'gpu':True
        }
        parameters_output = {'store_config': not predict}
        model = {
            'repository':os.path.join(self.models_dir, model_name),
            'create_repository':True
        }
        try:
            creat = self.dd.put_service(sname,model,'nbeats prediction','torch',parameters_input,parameters_mllib,parameters_output)
            self.log_progress(creat)
        except Exception:
            raise

    def train(self, data):
        parameters_input = {
            'db':False,
            'separator':',',
            'shuffle':True,
            'scale':True,
            'offset':self.offset,
            'ignore':self.ignored_cols,
            'forecast_timesteps':self.forecast,
            'backcast_timesteps':self.backcast
        }
        solver_params = {
            'snapshot':20000,
            'solver_type':'RANGER_PLUS',
            'test_initialization':False
        }
        solver_params.update(self.solver_params)

        parameters_mllib = {
            'net': {'batch_size':self.batch_size,'test_batch_size':min(self.batch_size, 10)},
            'solver': solver_params
        }
        parameters_output = {
            'measure':['L1']
        }
        train_response = self.dd.post_train(self.sname, data, parameters_input, parameters_mllib, parameters_output, jasync=True)

    def get_timeserie_results_nbeats(self, data):
        """
        run predict on nbeats service
        """
        parameters_input = {
            'connector':'csvts',
            'separator':',',
            'scale':True,
            'db':False,
            'backcast_timesteps':self.backcast,
            'forecast_timesteps':self.forecast
        }
        parameters_mllib = {'net':{'test_batch_size':1}}
        parameters_output = {}
        res = self.dd.post_predict(self.get_predict_sname(),data,parameters_input,parameters_mllib,parameters_output)
        if res["status"]["code"] != 200:
            print(res)
        return res

    def get_preds_targets_nbeats(self, datafile):
        timesteps = self.backcast + self.forecast

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
            if l not in self.ignored_cols:
                col_labels.append(get_col(header,l))

        npoints = num_lines-self.backcast
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

        for nline in self.progress_bar(range(0,num_lines-timesteps + self.pred_interval, self.pred_interval)): # sliding window
            # take last iteration into account
            it_points = min(self.forecast, num_lines - self.backcast - nline)

            if it_points <= 0:
                continue
            # input
            # csvdata = io.StringIO()
            # datawriter = csv.writer(csvdata)
            # for j in range(0,self.backcast):
            #    datapoint = raw_data[nline+j]
            #    datawriter.writerow(datapoint)
            data = "".join(raw_lines[nline:nline + self.backcast])

            # output => forecast
            targets[nline:nline + it_points] = raw_data[nline+self.backcast:nline + self.backcast + it_points,col_labels]

            out = self.get_timeserie_results_nbeats([csvheader.getvalue(), data])
            out = self.get_dd_predictions(out)

            pred = np.array([out[0]['series'][i]['out'] for i in range(it_points)], dtype=np.double)
            predictions[nline:nline + self.forecast] += pred
            npreds[nline:nline+self.forecast] += 1

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
        timesteps = self.backcast + self.forecast

        with open(datafile) as linefile:
            raw_lines = linefile.readlines()[1:]
            num_lines = len(raw_lines)

        if start < self.backcast:
            raise ValueError("start %d < self.backcast %d" % (start, self.backcast))

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
            if l not in self.ignored_cols:
                col_labels.append(get_col(header,l))

        npoints = stop - start
        ntarg = min(npoints, num_lines - self.backcast)
        nlabels = len(col_labels)

        csvheader=io.StringIO()
        headerwriter = csv.writer(csvheader)
        headerwriter.writerow(header)

        predictions = np.full((npoints,nlabels),0, dtype=np.double)
        targets = np.full((ntarg,nlabels),0, dtype=np.double)

        prev = raw_data[start-self.backcast:start,:]

        for nline in self.progress_bar(range(start, stop, self.forecast)):
            it_points = min(self.forecast, stop - nline)
            pred_line = nline - start

            # input
            csvdata = io.StringIO()
            datawriter = csv.writer(csvdata)
            for j in range(0,self.backcast):
                datapoint = prev[- self.backcast + j]
                datawriter.writerow(datapoint)

            # output => forecast
            targ_points = min(self.forecast, num_lines - nline)
            if nline + targ_points <= num_lines:
                targets[pred_line:pred_line + targ_points] = raw_data[nline:nline + targ_points,col_labels]

            out = self.get_timeserie_results_nbeats([csvheader.getvalue(),csvdata.getvalue()])
            out = self.get_dd_predictions(out)

            pred = np.array([out[0]['series'][i]['out'] for i in range(it_points)], dtype=np.double)
            predictions[pred_line:pred_line + it_points] = pred

            # Update data signal with prediction (autoregressive)
            if it_points == self.forecast: # no need to update for the last iteration
                prev = np.concatenate(
                        (prev[- self.backcast + self.forecast:],
                        np.full((self.forecast,prev.shape[1]), 0, dtype=np.double))
                    )
                prev[-self.forecast:,col_labels] = predictions[pred_line: pred_line + self.forecast]

        return predictions, targets

    def predict_all(self, override = False):
        if not override:
            self.load_targets()
            self.load_preds_errors()

        # nbeats target skip first ts (backcast distance)
        self.targs = {i: self.targs[i][self.backcast:,] for i in self.targs}

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

                if self.autoregressive:
                    pred, targ = self.get_nbeats_autoregression(datapath, self.backcast, -1)
                else:
                    pred, predictions_list, npreds, targ = self.get_preds_targets_nbeats(datapath)

                self.preds[datafile][model] = pred
                common_len = min(len(pred), len(targ))
                self.errors[datafile][model] = pred[:common_len] - targ[:common_len]
                self.targs[datafile] = targ

            predicted_models.append(model)
            self.delete_service(predict = True)

        self.dump_model_preds(predicted_models)
        self.log_job_done()
