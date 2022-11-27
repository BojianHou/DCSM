"""
This module provides several helper functions for model training and testing.

"""

import numpy as np
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from scipy.stats import linregress
from scipy import stats
from scipy.stats import weibull_min
from models.dcsm_api import DeepClusteringSurvivalMachines


def combine_t_e(t, e):  # t is for time and e is for event (indicator)
    # this function combines t and e into y, like a list of tuple
    y = np.zeros(len(t), dtype=[('cens', '?'), ('time', 'f')])
    for idx, item in enumerate(t):
        if e[idx] == 0:
            y[idx] = (False, item)
        else:
            y[idx] = (True, item)
    return y


def train_test_DCSM(param, X_train, X_test, y_train, y_test, seed=420, fix=False, method='DCSM'):
    print('param: ', param)
    e_train = np.array([[item[0] * 1 for item in y_train]]).T
    t_train = np.array([[item[1] for item in y_train]]).T
    e_test = np.array([[item[0] * 1 for item in y_test]]).T
    t_test = np.array([[item[1] for item in y_test]]).T

    model = DeepClusteringSurvivalMachines(k=param['k'], fix=fix, distribution=param['distribution'],
                 layers=param['layers'], discount=param['discount'],
                 random_state=seed, is_seed=True)

    print('method: ', method)

    # The fit method is called to train the model
    model.fit([X_train, X_test], [t_train, t_test], [e_train, e_test],
              iters=param['iters'], learning_rate=param['learning_rate'])
    processed_data = model._preprocess_training_data(X_train, t_train, e_train,
                                                     vsize=0.15, val_data=None, random_state=100)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data
    # calculate the C index
    pred_train = model.predict_risk(x_train, t_train.max())
    pred_val = model.predict_risk(x_val, t_val.max())
    # make sure there is no nan, inf and -inf in the prediction
    pred_train = np.nan_to_num(pred_train, nan=0, posinf=0, neginf=0)
    pred_val = np.nan_to_num(pred_val, nan=0, posinf=0, neginf=0)
    c_index = concordance_index_censored([True if i == 1 else False for i in e_train],
                                         t_train[:, 0].detach().cpu(), pred_train[:, 0])[0]
    print('c-index on the training data: {:.4f}'.format(c_index))
    c_index = concordance_index_censored([True if i == 1 else False for i in e_val],
                                         t_val[:, 0].detach().cpu(), pred_val[:, 0])[0]
    print('c-index on the validation data: {:.4f}'.format(c_index))

    # calculate the C index
    pred = model.predict_risk(X_test, t_test.max())
    # make sure there is no nan, inf and -inf in the prediction
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    c_index = concordance_index_censored([True if i == 1 else False for i in e_test],
                                         t_test[:, 0], pred[:, 0])[0]
    print('c-index on the testing data: {:.4f}'.format(c_index))

    pred_time = model.predict_mean(X_test).reshape(-1,1)
    rae_nc = rae(pred_time[e_test == 1], t_test[e_test == 1], 1 - e_test[e_test == 1])
    rae_c = rae(pred_time[e_test == 0], t_test[e_test == 0], 1 - e_test[e_test == 0])

    return model, c_index, pred, pred_time, rae_nc, rae_c


def test_DCSM(model, X_test, y_test):

    e_test = np.array([[item[0] * 1 for item in y_test]]).T
    t_test = np.array([[item[1] for item in y_test]]).T

    # calculate the C index
    pred = model.predict_risk(X_test, t_test.max())
    # make sure there is no nan, inf and -inf in the prediction
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    c_index = concordance_index_censored([True if i == 1 else False for i in e_test],
                                         t_test[:, 0], pred[:, 0])[0]
    print('c-index on the testing data: {:.4f}'.format(c_index))

    pred_time = model.predict_mean(X_test).reshape(-1,1)
    rae_nc = rae(pred_time[e_test == 1], t_test[e_test == 1], 1 - e_test[e_test == 1])
    rae_c = rae(pred_time[e_test == 0], t_test[e_test == 0], 1 - e_test[e_test == 0])

    return c_index, pred, pred_time, rae_nc, rae_c


def sample_weibull(scales, shape, n_samples=200):
    scales[scales <= 0] = 1e-5
    scales = np.nan_to_num(scales, nan=1e-5, posinf=1e-5, neginf=1e-5)
    shape = np.nan_to_num(shape, nan=1e-5, posinf=1e-5, neginf=1e-5)
    return np.transpose(weibull_min.rvs(shape, loc=0, scale=scales, size=(n_samples, scales.shape[0])))


def rae(t_pred, t_true, cens_t):
    # Relative absolute error as implemented by Chapfuwa et al.
    abs_error_i = np.abs(t_pred - t_true)
    pred_great_empirical = t_pred > t_true
    min_rea_i = np.minimum(np.divide(abs_error_i, t_true + 1e-8), 1.0)
    idx_cond = np.logical_and(cens_t, pred_great_empirical)
    min_rea_i[idx_cond] = 0.0

    return np.sum(min_rea_i) / len(t_true)


def calibration(predicted_samples, t, d):
    predicted_samples = np.nan_to_num(predicted_samples, nan=1e-5, posinf=1e-5, neginf=1e-5)
    t = np.nan_to_num(t, nan=0, posinf=0, neginf=0)
    d = np.nan_to_num(d, nan=0, posinf=0, neginf=0)

    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=d)

    range_quant = np.arange(start=0, stop=1.010, step=0.010)
    t_empirical_range = np.unique(np.sort(np.append(t, [0])))
    km_pred_alive_prob = [kmf.predict(i) for i in t_empirical_range]
    empirical_dead = 1 - np.array(km_pred_alive_prob)

    km_dead_dist, km_var_dist, km_dist_ci = compute_km_dist(predicted_samples, t_empirical_range=t_empirical_range,
                                                            event=d)

    slope, intercept, r_value, p_value, std_err = linregress(x=km_dead_dist, y=empirical_dead)

    return slope


def compute_km_dist(predicted_samples, t_empirical_range, event):
    km_dead = []
    km_surv = 1

    km_var = []
    km_ci = []
    km_sum = 0

    kernel = []
    e_event = event

    for j in np.arange(len(t_empirical_range)):
        r = t_empirical_range[j]
        low = 0 if j == 0 else t_empirical_range[j - 1]
        area = 0
        censored = 0
        dead = 0
        at_risk = len(predicted_samples)
        count_death = 0
        for i in np.arange(len(predicted_samples)):
            e = e_event[i]
            if len(kernel) != len(predicted_samples):
                # print('predicted_samples {}: {}'.format(i, predicted_samples[i]))
                kernel_i = stats.gaussian_kde(predicted_samples[i])
                kernel.append(kernel_i)
            else:
                kernel_i = kernel[i]
            at_risk = at_risk - kernel_i.integrate_box_1d(low=0, high=low)

            if e == 1:
                count_death += kernel_i.integrate_box_1d(low=low, high=r)
        if at_risk == 0:
            break
        km_int_surv = 1 - count_death / at_risk
        km_int_sum = count_death / (at_risk * (at_risk - count_death))

        km_surv = km_surv * km_int_surv
        km_sum = km_sum + km_int_sum

        km_ci.append(ci_bounds(cumulative_sq_=km_sum, surv_t=km_surv))

        km_dead.append(1 - km_surv)
        km_var.append(km_surv * km_surv * km_sum)

    return np.array(km_dead), np.array(km_var), np.array(km_ci)


def ci_bounds(surv_t, cumulative_sq_, alpha=0.95):
    # print("surv_t: ", surv_t, "cumulative_sq_: ", cumulative_sq_)
    # This method calculates confidence intervals using the exponential Greenwood formula.
    # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
    # alpha = 0.95
    if surv_t > 0.999:
        surv_t = 1
        cumulative_sq_ = 0
    alpha = 0.95
    constant = 1e-8
    alpha2 = stats.norm.ppf((1. + alpha) / 2.)
    v = np.log(surv_t)
    left_ci = np.log(-v)
    right_ci = alpha2 * np.sqrt(cumulative_sq_) * 1 / v

    c_plus = left_ci + right_ci
    c_neg = left_ci - right_ci

    ci_lower = np.exp(-np.exp(c_plus))
    ci_upper = np.exp(-np.exp(c_neg))

    return [ci_lower, ci_upper]

