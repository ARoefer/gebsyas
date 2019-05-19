#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


from gebsyas.ml.feature_model         import *
from gebsyas.ml.test.sample_generator import generate_shelf_samples
from gebsyas.core.subs_ds             import Structure, ListStructure
from gebsyas.utils                    import bb
from giskardpy.symengine_wrappers     import inverse_frame

if __name__ == '__main__':
    observations = generate_shelf_samples(1000)

    pmodel = PModel(['none', 'prismatic', 'hinge'])

    flat_obs = []

    device = torch.device('cpu')

    for o in observations:
        obs_dict = o['s'].to_flat_dict('static/')
        obs_dict.update(o['d'].to_flat_dict('dynamic/'))
        obs_dict['relT'] = inverse_frame(o['s'].pose) * o['d'].pose
        pmodel.find_features([obs_dict])
        flat_obs.append((o['k'], obs_dict))

    pmodel.print_feature_info()
    pmodel.train_model(flat_obs[:int(len(flat_obs) * 0.8)], 1000)
    #model = pmodel.build_model()



    # learning_rate = 0.001
    # training_in   = torch.from_numpy(np.vstack([pmodel.build_input_vector(o) for _, o in flat_obs[:int(len(flat_obs) * 0.8)]]))
    # training_out  = torch.from_numpy(np.vstack([pmodel.build_output_vector(k) for k, _ in flat_obs[:int(len(flat_obs) * 0.8)]]))

    # print(model)
    # print(training_in)
    # print(training_out)


    # last_loss = 1e5
    # loss_ot  = [last_loss]
    # dloss_ot = []
    # best_model = (last_loss, -1, None)
    # best_model_switch = []
    # for t in range(1000):
    #     out_prediction = training_in.mm(model).clamp(min=0).renorm(1,0,1)
    #     #print(out_prediction)

    #     loss = (out_prediction - training_out).pow(2).sum()
    #     print(t, loss.item())

    #     if loss.item() < best_model[0]:
    #         best_model = (loss.item(), t, model.clone())
    #         best_model_switch.append(t)

    #     loss.backward()

    #     with torch.no_grad():
    #         #print(model.grad)
    #         model -= learning_rate * model.grad

    #         model.grad.zero_()

    #     if abs(best_model[0] - loss.item()) > best_model[0] * np.exp(-0.5 * (t - best_model[1])):
    #         break
    #     last_loss = loss.item()
    #     dloss_ot.append(last_loss - loss_ot[-1])
    #     loss_ot.append(last_loss)

    # model = best_model[2]

    # eval_in  = torch.from_numpy(np.vstack([pmodel.build_input_vector(o) for _, o in flat_obs[:-int(len(flat_obs) * 0.2)]]))

    # eval_out = torch.from_numpy(np.vstack([pmodel.build_output_vector(k) for k, _ in flat_obs[:-int(len(flat_obs) * 0.2)]]))

    # print('Eval in:\n{}\nEval out:\n{}'.format(eval_in, eval_out))

    # eval_predicition = eval_in.mm(model).clamp(min=0).renorm(1,0,1)
    
    # with torch.no_grad():
    #     loss = sum([min(1, abs(x)) for x in (eval_predicition.max(1)[1] - eval_out.max(1)[1])]).item() / float(eval_out.size(0))
        
    #     print('Eval prediction:\n{}\nFinal loss:\n{}'.format(eval_predicition, loss))#.item()))
    #     #print('Model:\n{}'.format(model))

    # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5,6))
    # ax1.plot(loss_ot[1:], 'r', label='Loss')
    # #ax1.plot(best_model_switch, best_model_value, 'gx', label='Better Model Discovered')
    # ax1.legend(loc='center right')
    # ax1.set_title('Loss')
    # ax2.plot(dloss_ot[1:], 'b', label=r'$\Delta Loss$')
    # ax2.legend(loc='lower right')
    # ax2.set_title(r'$\Delta Loss$')
    # for x in best_model_switch:
    #     ax1.axvline(x=x, linewidth=0.5, ymin=0.9)
    #     ax2.axvline(x=x, linewidth=0.5, ymin=0.9)
    # fig.tight_layout()
    # fig.savefig('loss_plot.png')