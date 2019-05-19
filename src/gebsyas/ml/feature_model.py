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

COLORS = ['g', 'c', 'm', 'y', 'k', 'w']

class TrainingReport(object):
    def __init__(self, title):
        super(TrainingReport, self).__init__()
        self.title = title
        self.loss  = []
        self.dloss = [0.0]
        self._iteration = 0
        self.events = {}

    def log_loss(self, loss):
        if len(self.loss) > 0:
            self.dloss.append(loss - self.loss[-1])
        self.loss.append(loss)
        self._iteration += 1

    def log_event(self, event, *args):
        if event not in self.events:
            self.events[event] = []
        self.events[event].append(tuple((self._iteration,) + args))

    def plot(self, ax):
        cindex  = 0
        patches = [ax.plot(self.loss, 'r', label='Loss')[0], 
                   ax.plot(self.dloss, 'b', label=r'$\Delta Loss$')[0]]
        for e, t in self.events.items():
            patches.append(mlines.Line2D([], [], color=COLORS[cindex % len(COLORS)],  label=e))
            for x in t:
                ax.axvline(x=x[0], linewidth=0.5, ymin=0.9, color=COLORS[cindex % len(COLORS)])
        ax.legend(handles=patches, loc='center right')
        ax.set_title('{} Loss'.format(self.title))


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
                        self.features[k] = NonOrderedFeature()
                    elif DLScalar.is_a(v):
                        self.features[k] = OrderedFeature()
                    elif DLVector.is_a(v) or DLPoint.is_a(v):
                        self.features[k] = Vector3dFeature()
                    elif DLTransform.is_a(v):
                        self.features[k] = TransformFeature()
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
            training_in  = torch.from_numpy(np.vstack([f.to_nn_input(o[k]) for _, o in observations if k in o]))
            training_out = torch.from_numpy(np.vstack([self.build_output_vector(v) for v, o in observations if k in o]))

            feature_model = f.generate_classifier_tensor(len(self.labels))

            best_model = None
            for t in range(max_iterations):
                out_prediction = training_in.mm(feature_model).clamp(min=0).renorm(1,0,1)
                #print(out_prediction)

                loss = (out_prediction - training_out).pow(2).sum()
                report.log_feature_loss(k, loss.item())

                if best_model is None:
                    best_model = (loss.item(), t, feature_model.clone())
                elif loss.item() < best_model[0]:
                    best_model = (loss.item(), t, feature_model.clone())
                    report.log_feature_event(k, 'model switch')

                loss.backward()

                with torch.no_grad():
                    feature_model.sub_(learning_rate * feature_model.grad)
                    feature_model.grad.zero_()

                if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
                    break

            f.c_tensor = best_model[2]
            with torch.no_grad():
                results.append((training_in.mm(f.c_tensor).clamp(min=0).renorm(1,0,1)).numpy())

        self.c_mix_tensor = torch.ones(sum([1 for f in self.features.values() if f.observation_coeff() == 1.0]) * len(self.labels), len(self.labels), dtype=torch.float64, requires_grad=True)
        training_in = torch.from_numpy(np.hstack([results[x] for x in range(len(self.features)) if self.features.values()[x].observation_coeff() == 1.0]))
        training_out = torch.from_numpy(np.vstack([self.build_output_vector(v) for v, o in observations]))

        best_model = None
        for t in range(max_iterations):
            out_prediction = training_in.mm(self.c_mix_tensor).clamp(min=0).renorm(1,0,1)
            #print(out_prediction)

            loss = (out_prediction - training_out).pow(2).sum()
            report.log_loss(loss.item())

            if best_model is None:
                best_model = (loss.item(), t, self.c_mix_tensor.clone())
            elif loss.item() < best_model[0]:
                best_model = (loss.item(), t, self.c_mix_tensor.clone())
                report.log_event('model switch')

            loss.backward()

            with torch.no_grad():
                self.c_mix_tensor.sub_(learning_rate * self.c_mix_tensor.grad)
                self.c_mix_tensor.grad.zero_()

            if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
                break

        self.c_mix_tensor = best_model[0]        

        fig = report.create_figure()
        fig.savefig('kpm_loss_plots.png')

    def build_model(self):
        inputs = self.features.values()
        in_width = sum([i.nn_num_inputs() for i in inputs])
        return torch.ones(in_width, len(self.labels), dtype=torch.float64, requires_grad=True)

    def build_input_vector(self, d):
        return torch.from_numpy(np.hstack([i.to_nn_input(d[k]) if k in d else np.zeros(i.nn_num_inputs()) for k, i in self.features.items()]))

    def build_output_vector(self, value):
        return torch.tensor([1.0 if value == l else 0.0 for l in self.labels], dtype=torch.float64)

    def print_feature_info(self):
        print('\n'.join(['{}:\n  Observation coeff: {}'.format(k, f.observation_coeff()) for k, f in self.features.items()]))


class Feature(object):
    def __init__(self, n_labels, obs_count=0, total_observations=0):
        super(Feature, self).__init__()
        self.observation_count  = obs_count
        self.total_observations = total_observations
        self.c_tensor = None
        self.n_labels = n_labels

    def process_non_observation(self):
        self.total_observations += 1

    def process_observation(self, value):
        self.observation_count  += 1
        self.total_observations += 1

    def observation_coeff(self):
        return self.observation_count / float(self.total_observations) if self.total_observations > 0 else 0

    def generate_classifier_tensor(self, n_labels):
        if self.c_tensor is None or self.c_tensor.size(0) != n_labels:
            self.c_tensor = torch.ones(self.nn_num_inputs(), n_labels, dtype=torch.float64, requires_grad=True)
        return self.c_tensor

    def nn_num_inputs(self):
        raise NotImplementedError

    def hl_forward(self, value):
        return self.forward(self.to_nn_input(value))

    def forward(self, value):
        raise NotImplementedError

    def to_nn_input(self, value):
        raise NotImplementedError


class OrderedFeature(Feature):
    def __init__(self, n_labels, obs_count=0, total_observations=0, dim=2):
        super(OrderedFeature, self).__init__(n_labels, obs_count, total_observations)
        self.mean     = torch.zeros(dim, n_labels, dtype=torch.float64, requires_grad=True)
        self.variance =  torch.ones(dim, n_labels, dtype=torch.float64, requires_grad=True)
        self.dim      = dim

    def nn_num_inputs(self):
        return self.dim

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = np.array([value, abs(value)], dtype=float)
        #print(out, out.dtype)
        return out

    def train_model(self, observations, result_gen, learning_rate=0.01, max_iterations=100):
        training_in  = torch.from_numpy(np.vstack([f.to_nn_input(o) for _, o in observations]))
        training_out = torch.from_numpy(np.vstack([result_gen(v) for v, _ in observations]))

    def forward(self, value):
        return torch.ones().div(self.variance.pow(2)).exp(value.sub(self.mean).pow(2).div(self.variance.pow(2)*2))


class Vector3dFeature(Feature):
    def __init__(self):
        super(VectorFeature, self).__init__()
        self.input_transform = np.vstack((np.eye(3), np.ones(3)))

    def nn_num_inputs(self):
        return self.input_transform.shape[0]

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = self.input_transform.dot(np.array([value[0], value[1], value[2]], dtype=float))
        #print(out, out.dtype)
        return out


class TransformFeature(Feature):
    def __init__(self):
        super(TransformFeature, self).__init__()
        self.input_transform = np.vstack((np.eye(3), np.ones(3)))

    def nn_num_inputs(self):
        return 7

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        axis, angle = axis_angle_from_matrix(value)
        pos = pos_of(value)
        out = np.array([axis[0], axis[1], axis[2], angle, pos[0], pos[1], pos[2]], dtype=float)
        #print(out, out.dtype)
        return out


class NonOrderedFeature(Feature):
    def __init__(self):
        super(NonOrderedFeature, self).__init__()
        self.values = {}

    def process_observation(self, value):
        super(NonOrderedFeature, self).process_observation(value)
        if value not in self.values:
            self.values[value] = len(self.values)

    def nn_num_inputs(self):
        return len(self.values)

    def to_nn_input(self, value):
        #print('To nn input {}:\n{}'.format(type(self), value))
        out = np.zeros(len(self.values), dtype=float)
        if value in self.values:
            out[self.values[value]] = 1.0
        #print(out, out.dtype)
        return out