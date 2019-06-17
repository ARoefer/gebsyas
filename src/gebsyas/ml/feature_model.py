import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines

from collections import OrderedDict
from giskardpy.symengine_wrappers import axis_angle_from_matrix, pos_of
from gebsyas.core.dl_types import DLScalar, DLVector, DLPoint, DLTransform
from gebsyas.core.subs_ds  import Structure, ListStructure
from gebsyas.utils         import bb
from gebsyas.plotting      import ValueRecorder

COLORS = ['g', 'c', 'm', 'y', 'k', 'w']

class TrainingReport(ValueRecorder):
    def __init__(self, title):
        super(TrainingReport, self).__init__('{} Loss'.format(self.title), 'Loss', r'\$Delta Loss')
        self.log_data(r'\$Delta Loss', 0.0)
        self._iteration = 0
        self.events = {}

    def log_loss(self, loss):
        if len(self.data['Loss']) > 0:
            self.log_data(r'\$Delta Loss', loss - self.data['Loss'][-1])
        self.log_data('Loss', loss)
        self._iteration += 1

    def log_event(self, event, *args):
        if event not in self.events:
            self.events[event] = []
        self.events[event].append(tuple((self._iteration,) + args))

    def plot(self, ax):
        super(TrainingReport, self).plot(ax)
        for e, t in self.events.items():
            self.patches.append(mlines.Line2D([], [], color=COLORS[cindex % len(COLORS)],  label=e))
            for x in t:
                ax.axvline(x=x[0], linewidth=0.5, ymin=0.9, color=COLORS[cindex % len(COLORS)])
            cindex += 1
        ax.legend(handles=self.patches, loc='center right')


class ModelTrainingReport(TrainingReport):
    def __init__(self, title):
        super(ModelTrainingReport, self).__init__(title)
        self.feature_report   = {}
        self.feature_timeline = {}

    def log_feature_loss(self, feature, loss):
        if feature not in self.feature_report:
            self.feature_report[feature]   = TrainingReport(feature)
            self.feature_timeline[feature] = self._iteration
        self.feature_report[feature].log_loss(loss)

    def log_feature_event(self, feature, event, *args):
        if feature not in self.feature_report:
            self.feature_report[feature]   = TrainingReport(feature)
            self.feature_timeline[feature] = self._iteration
        self.feature_report[feature].log_event(event, args)        

    def create_figure(self, summary_width=2, summary_height=1):
        rows = max(summary_width, int(math.ceil(math.sqrt(len(self.feature_report)))))
        cols = int(math.ceil((len(self.feature_report) + summary_width * summary_height) / float(rows)))

        gridsize     = (rows, cols)
        plot_size    = (3, 2)
        fig          = plt.figure(figsize=(cols*plot_size[0], rows*plot_size[1]))
        summary_axes = plt.subplot2grid(gridsize, (0, 0), colspan=summary_width, rowspan=summary_height)
        indices = [(x / cols, x % cols) for x in range(summary_width * summary_height + len(self.feature_report)) if x / cols >= summary_height or x % cols >= summary_width]
        axes    = [plt.subplot2grid(gridsize, x, colspan=1, rowspan=1) for x in indices]
        self.plot(summary_axes)
        for x, f in enumerate(self.feature_report.values()):
            f.plot(axes[x])
        fig.tight_layout()
        return fig

    def add_report(self, feature, report):
        self.feature_report[feature] = report


class PModel(object):
    def __init__(self, labels):
        self.labels = labels
        self.features = OrderedDict()

    def find_features(self, observations):
        for obs in observations:
            for k, v in obs.items():
                t_v = type(v)
                if k not in self.features:
                    if t_v == str:
                        self.features[k] = NonOrderedFeature(len(self.labels))
                    elif DLScalar.is_a(v):
                        self.features[k] = OrderedFeature(len(self.labels))
                    elif DLVector.is_a(v) or DLPoint.is_a(v):
                        self.features[k] = Vector3dFeature(len(self.labels))
                    elif DLTransform.is_a(v):
                        self.features[k] = TransformFeature(len(self.labels))
                    else:
                        continue
                self.features[k].process_observation(v)
        
        total_obs = max([f.total_observations for f in self.features.values()])
        for f in self.features.values():
            f.total_observations = total_obs

    def train_model(self, observations, learning_rate=0.01, max_iterations=100):
        report = ModelTrainingReport('Kinematic Prediction Model')
        results = []
        for k, f in self.features.items():
            print('Training feature classifier for feature "{}"'.format(k))
            
            f_report = f.train_model([(v, o[k]) for v, o in observations if k in o], self.build_output_vector, learning_rate, max_iterations)
            f_report.title = k
            report.add_report(k, f_report)

        # self.c_mix_tensor = torch.ones(sum([1 for f in self.features.values() if f.observation_coeff() == 1.0]) * len(self.labels), len(self.labels), dtype=torch.float64, requires_grad=True)
        # training_in = torch.from_numpy(np.hstack([results[x] for x in range(len(self.features)) if self.features.values()[x].observation_coeff() == 1.0]))
        # training_out = torch.from_numpy(np.vstack([self.build_output_vector(v) for v, o in observations]))

        # best_model = None
        # for t in range(max_iterations):
        #     out_prediction = training_in.mm(self.c_mix_tensor).clamp(min=0).renorm(1,0,1)
        #     #print(out_prediction)

        #     loss = (out_prediction - training_out).pow(2).sum()
        #     report.log_loss(loss.item())

        #     if best_model is None:
        #         best_model = (loss.item(), t, self.c_mix_tensor.clone())
        #     elif loss.item() < best_model[0]:
        #         best_model = (loss.item(), t, self.c_mix_tensor.clone())
        #         report.log_event('model switch')

        #     loss.backward()

        #     with torch.no_grad():
        #         self.c_mix_tensor.sub_(learning_rate * self.c_mix_tensor.grad)
        #         self.c_mix_tensor.grad.zero_()

        #     if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
        #         break

        # self.c_mix_tensor = best_model[0]        

        fig = report.create_figure()
        fig.savefig('kpm_loss_plots.png')

    def build_model(self):
        inputs = self.features.values()
        in_width = sum([i.nn_num_inputs() for i in inputs])
        return torch.ones(in_width, len(self.labels), dtype=torch.float64, requires_grad=True)

    def build_input_vector(self, d):
        return torch.from_numpy(np.vstack([i.to_nn_input(d[k]) if k in d else np.zeros((i.nn_num_inputs(),1)) for k, i in self.features.items()]))

    def build_output_vector(self, value):
        return torch.tensor([[1.0 if value == l else 0.0 for l in self.labels]], dtype=torch.float64).transpose(1,0)

    def print_feature_info(self):
        print('\n'.join(['{}:\n  Observation coeff: {}'.format(k, f.observation_coeff()) for k, f in self.features.items()]))


class Feature(object):
    def __init__(self, n_labels, obs_count=0, total_observations=0):
        super(Feature, self).__init__()
        self.observation_count  = obs_count
        self.total_observations = total_observations
        self.n_labels = n_labels

    def process_non_observation(self):
        self.total_observations += 1

    def process_observation(self, value):
        self.observation_count  += 1
        self.total_observations += 1

    def observation_coeff(self):
        return self.observation_count / float(self.total_observations) if self.total_observations > 0 else 0

    def nn_num_inputs(self):
        raise NotImplementedError

    def hl_forward(self, value):
        return self.forward(self.to_nn_input(value))

    def forward(self, value):
        raise NotImplementedError

    def to_nn_input(self, value):
        raise NotImplementedError

    def train_model(self, observations, result_gen, learning_rate=0.01, max_iterations=100):
        raise NotImplementedError


class OrderedFeature(Feature):
    def __init__(self, n_labels, obs_count=0, total_observations=0, dim=2):
        super(OrderedFeature, self).__init__(n_labels, obs_count, total_observations)
        self.mean     = torch.zeros(dim, n_labels, 1, dtype=torch.float64, requires_grad=True)
        self.variance = torch.ones( dim, n_labels, 1, dtype=torch.float64, requires_grad=True)
        self.dim      = dim

    def nn_num_inputs(self):
        return self.dim

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = np.array([[value, abs(value)]], dtype=float).T
        #print(out, out.dtype)
        return out

    def forward(self, value):
        value = value.reshape(self.dim, 1, value.size(1))
        var   = 1.0 / self.variance.pow(2)
        e_var = -var.pow(2).reshape(var.size(0), var.size(1), 1)
        exponents  = (self.mean - value) * e_var
        potentials = (var * exponents.exp())
        return potentials.sum(0).reshape(self.n_labels, value.size(2))

    def train_model(self, observations, result_gen, learning_rate=0.01, max_iterations=100):
        if len(observations) == 0:
            raise Exception('Feature can not be trained on empty observation set!')

        training_in  = torch.from_numpy(np.hstack([self.to_nn_input(o) for _, o in observations]))
        training_out = torch.from_numpy(np.hstack([result_gen(v) for v, _ in observations]))

        true_error_mask = training_out.max(0)[1]

        report = TrainingReport('')

        best_model = None
        for t in range(max_iterations):
            out_prediction = self.forward(training_in)
            #print(out_prediction)

            loss = (out_prediction - training_out).pow(2).sum()

            if best_model is None:
                best_model = (loss.item(), t, self.mean.clone(), self.variance.clone())
            elif loss.item() < best_model[0]:
                best_model = (loss.item(), t, self.mean.clone(), self.variance.clone())
                report.log_event('model switch')

            loss.backward()

            with torch.no_grad():
                report.log_loss((out_prediction.max(0)[1] - true_error_mask).abs().clamp(0,1).sum().item() / len(observations))
                self.mean.sub_(learning_rate * self.mean.grad)
                self.variance.sub_(learning_rate * self.variance.grad)
                self.mean.grad.zero_()
                self.variance.grad.zero_()

            if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
                break

        self.mean     = best_model[2]
        self.variance = best_model[3]
        return report


class Vector3dFeature(OrderedFeature):
    def __init__(self, n_labels, obs_count=0, total_observations=0):
        super(VectorFeature, self).__init__(n_labels, obs_count, total_observations, 4)
        self.input_transform = np.vstack((np.eye(3), np.ones(3)))

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = self.input_transform.dot(np.array([[value[0], value[1], value[2]]], dtype=float).T)
        #print(out, out.dtype)
        return out


class TransformFeature(OrderedFeature):
    def __init__(self, n_labels, obs_count=0, total_observations=0):
        super(TransformFeature, self).__init__(n_labels, obs_count, total_observations, 7)

    def nn_num_inputs(self):
        return 7

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        axis, angle = axis_angle_from_matrix(value)
        pos = pos_of(value)
        return np.array([[axis[0], axis[1], axis[2], angle, pos[0], pos[1], pos[2]]], dtype=float).T



class NonOrderedFeature(Feature):
    def __init__(self, n_labels, obs_count=0, total_observations=0):
        super(NonOrderedFeature, self).__init__(n_labels, obs_count, total_observations)
        self.values  = {}
        self.c_model = None

    def process_observation(self, value):
        super(NonOrderedFeature, self).process_observation(value)
        if value not in self.values:
            self.values[value] = len(self.values)

    def nn_num_inputs(self):
        return len(self.values)

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = np.zeros((len(self.values), 1), dtype=float)
        if value in self.values:
            out[self.values[value]] = 1.0
        #print(out, out.dtype)
        return out

    def forward(self, value):
        if self.c_model is None:
            self.c_model = torch.ones(self.n_labels, len(self.values), dtype=torch.float64, requires_grad=True)
        return self.c_model.mm(value).clamp(min=0).renorm(1,0,1)

    def train_model(self, observations, result_gen, learning_rate=0.01, max_iterations=100):
        training_in  = torch.from_numpy(np.hstack([self.to_nn_input(o) for _, o in observations]))
        training_out = torch.from_numpy(np.hstack([result_gen(v) for v, _ in observations]))

        print(training_out)

        true_error_mask = training_out.max(0)[1]

        report = TrainingReport('')

        best_model = None
        for t in range(max_iterations):
            out_prediction = self.forward(training_in)

            loss = (out_prediction - training_out).pow(2).sum()

            if best_model is None:
                best_model = (loss.item(), t, self.c_model.clone())
            elif loss.item() < best_model[0]:
                best_model = (loss.item(), t, self.c_model.clone())
                report.log_event('model switch')

            loss.backward()

            with torch.no_grad():
                report.log_loss(loss.item()) #(out_prediction.max(0)[1] - true_error_mask).abs().clamp(0,1).sum().item() / len(observations))
                self.c_model.sub_(learning_rate * self.c_model.grad)
                self.c_model.grad.zero_()

            if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
                break

        self.c_model = best_model[2]
        return report