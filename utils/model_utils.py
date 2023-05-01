"""
This module provides several functions for model training utilities.
"""

from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from models.dcsm_torch import DeepClusteringSurvivalMachinesTorch
from .losses import unconditional_loss, conditional_loss
from .losses import predict_cdf


def get_optimizer(model, lr):
    if model.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif model.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif model.optimizer == 'RMSProp':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer ' + model.optimizer +
                                  ' is not implemented')


def plot_loss_c_index(results_all, lr, bs, k, dist, discount):
    # plot the loss and the C Index
    fig, ax = plt.subplots()
    ax.plot(results_all[:, 0], color='tab:red', label='train loss')
    ax.plot(results_all[:, 1], color='tab:blue', label='test loss')
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(results_all[:, 2], color='tab:green', label='C Index Test')
    ax2.plot(results_all[:, 3], color='tab:orange', label='C Index Train')
    ax2.set_ylabel("C Index", fontsize=14)
    ax2.plot(np.nan, color='tab:red', label='train loss')  # print an empty line to represent loss
    ax2.plot(np.nan, color='tab:blue', label='test loss')

    ax2.legend(loc=0)
    ax.grid()
    plt.title('lr: {:.2e}, k: {}, bs: {}, {}, discount: {}'.format(lr, k, bs, dist, discount))
    plt.show()
    plt.close()


def pretrain_dcsm(model, t_train, e_train, t_valid, e_valid,
                 n_iter=10000, lr=1e-2, thres=1e-4):
    premodel = DeepClusteringSurvivalMachinesTorch(1, 1,
                                         dist=model.dist,
                                         risks=model.risks,
                                         optimizer=model.optimizer,
                                         random_state=model.random_state  # fix=model.fix,
                                         # is_seed=model.is_seed
                                         )  # .cuda()

    premodel.cuda()
    premodel.double()

    optimizer = get_optimizer(premodel, lr)

    oldcost = float('inf')
    patience = 0
    costs = []

    print('\nPretraining...')
    for i in tqdm(range(n_iter)):

        optimizer.zero_grad()
        loss = 0
        for r in range(model.risks):
            loss += unconditional_loss(premodel, t_train, e_train, str(r + 1))
        loss.backward()
        optimizer.step()

        valid_loss = 0
        for r in range(model.risks):
            valid_loss += unconditional_loss(premodel, t_valid, e_valid, str(r + 1))
        valid_loss = valid_loss.detach().cpu().numpy()
        costs.append(valid_loss)
        # print(valid_loss)
        if np.abs(costs[-1] - oldcost) < thres:
            patience += 1
            if patience == 3:
                break
                print('pre train stops at {}th iteration'.format(i))
        oldcost = costs[-1]

    return premodel


def _reshape_tensor_with_nans(data):
    data = data.reshape(-1)
    return data[~torch.isnan(data)]


def predict_risks(model, x, t):
    # x = torch.from_numpy(x).double().cuda()
    if not isinstance(t, list):
        t = [t]
    scores = predict_cdf(model, x, t)
    return 1 - np.exp(np.array(scores)).T


def train_dcsm(model,
              x_train, t_train, e_train,
              x_valid, t_valid, e_valid,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100):
    """Function to train the torch instance of the model."""

    # For padded variable length sequences we first unroll the input and
    # mask out the padded nans.
    t_train_ = _reshape_tensor_with_nans(t_train).cuda()
    e_train_ = _reshape_tensor_with_nans(e_train).cuda()
    t_valid_ = _reshape_tensor_with_nans(t_valid).cuda()
    e_valid_ = _reshape_tensor_with_nans(e_valid).cuda()

    premodel = pretrain_dcsm(model,
                            t_train_,
                            e_train_,
                            t_valid_,
                            e_valid_,
                            n_iter=10000,
                            lr=1e-2,
                            thres=1e-4)

    # if model.fix == False:
    for r in range(model.risks):
        model.shape[str(r + 1)].data.fill_(float(premodel.shape[str(r + 1)]))
        model.scale[str(r + 1)].data.fill_(float(premodel.scale[str(r + 1)]))
#     else:
#         for r in range(model.risks):
#             # initialize using the pretrained shape and scale with a small perturbation,
#             # whose mean is zero, std is the 1/10 of the original value
#             model.shape[str(r + 1)].data = float(premodel.shape[str(r + 1)]) + \
#                                            torch.normal(mean=torch.zeros(model.k),
#                                                         std=np.abs(float(premodel.shape[str(r + 1)])) / 10)
#             model.scale[str(r + 1)].data = float(premodel.scale[str(r + 1)]) + \
#                                            torch.normal(mean=torch.zeros(model.k),
#                                                         std=np.abs(float(premodel.scale[str(r + 1)])) / 10)

    model.double()
    optimizer = get_optimizer(model, lr)
    nbatches = int(x_train.shape[0] / bs) + 1

    best_dic = []
    shape_list = []
    scale_list = []

    # the 1st column is train loss and the 2nd is validation loss 3rd is valid c_ind, 4th is train c_ind
    results_all = np.zeros((n_iter, 4))

    print('\nTraining...')
    for i in tqdm(range(n_iter)):
        for j in range(nbatches):

            xb = x_train[j * bs:(j + 1) * bs].cuda()
            tb = t_train[j * bs:(j + 1) * bs].cuda()
            eb = e_train[j * bs:(j + 1) * bs].cuda()

            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = 0
            for r in range(model.risks):
                loss += conditional_loss(model,
                                         xb,
                                         _reshape_tensor_with_nans(tb),
                                         _reshape_tensor_with_nans(eb),
                                         elbo=elbo,
                                         risk=str(r + 1))
            loss.backward()
            optimizer.step()

        shape_list.append(model.shape[str(r + 1)].data)
        scale_list.append(model.scale[str(r + 1)].data)

        valid_loss = 0
        for r in range(model.risks):
            valid_loss += conditional_loss(model,
                                           x_valid.cuda(),
                                           t_valid_,
                                           e_valid_,
                                           elbo=False,
                                           risk=str(r + 1))

        valid_loss = valid_loss.detach().cpu().numpy()
        dic = deepcopy(model.state_dict())

        pred_train = predict_risks(model, x_train, t_train_.max())
        pred_val = predict_risks(model, x_valid, t_valid_.max())

        # make sure there is no nan, inf and -inf in the prediction
        pred_train = np.nan_to_num(pred_train, nan=0, posinf=0, neginf=0)
        pred_val = np.nan_to_num(pred_val, nan=0, posinf=0, neginf=0)
        c_index_train = concordance_index_censored([True if i == 1 else False for i in e_train],
                                                   t_train[:, 0].detach().cpu(), pred_train[:, 0])[0]
        c_index_valid = concordance_index_censored([True if i == 1 else False for i in e_valid],
                                                   t_valid[:, 0].detach().cpu(), pred_val[:, 0])[0]
        # choose the model with best C index
        if best_dic:
            if c_index_valid >= best_dic[0]:
                best_dic = [c_index_valid, dic, i]
        else:
            best_dic = [c_index_valid, dic, i]

        results_all[i, 0] = loss.detach().cpu().numpy()
        results_all[i, 1] = valid_loss
        results_all[i, 2] = c_index_valid
        results_all[i, 3] = c_index_train

    plot_loss_c_index(results_all, lr, bs, model.k, model.dist, model.discount)

    model.load_state_dict(best_dic[1])
    print('best model is chosen from {}th epoch'.format(best_dic[2]))
    return model, i
