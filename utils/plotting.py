import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter


def plot_Weibull_cdf(t_horizon, shape, scale, data_name='sim', num_inst=1000, num_feat=200, seed=42):
    step = 100
    for i in range(len(shape)):
        k = shape[i]
        b = scale[i]
        s = np.zeros(step)
        t_space = np.linspace(0, t_horizon, step)
        for j in range(step):
            s[j] = np.exp(-(np.power(np.exp(b) * t_space[j], np.exp(k))))
        plt.plot(t_space, s, label='Expert Distribution {}'.format(i))
    plt.legend()
    # plt.title('Weibull CDF, Data: {}, Seed: {}'.format(data_name, seed))
    plt.title('Weibull CDF, Data: {}'.format(data_name), fontsize=16)
    if data_name == 'sim':
        plt.savefig('./Figures/Weibull_cdf_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(len(shape), data_name, num_inst, num_feat, seed))
    else:
        plt.savefig('./Figures/Weibull_cdf_#clusters{}_{}_seed{}.png'.
                    format(len(shape), data_name, seed))
    plt.show()
    plt.close()


def plot_loss_c_index(results_all, lr, epoch, bs):
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
    plt.title('lr: {:.2e}, epoch: {}, batch_size: {}'.format(lr, epoch, bs))
    plt.show()
    plt.close()


def visualize(X_train_list, X_test_list, data_name, is_normalize=0, is_TSNE=1):
    """This function is to visualize the scatter plot with clustering information"""

    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    X = np.concatenate((X_train, X_test), axis=0)
    # normalize
    if is_normalize == 1:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if is_TSNE == 1:
        # embed using TSNE
        embeddings = TSNE(random_state=42).fit_transform(X)
    else:
        # embed using UMAP
        trans = umap.UMAP(random_state=42).fit(X)
        embeddings = trans.embedding_
        # embeddings = []


    xlim = [-100, 95]
    ylim = [-90, 90]

    # show each cluster separately on all train data
    len_train = 0
    for idx, f in enumerate(X_train_list):
        plt.scatter(embeddings[len_train:(len_train + len(f)), 0],
                    embeddings[len_train:(len_train + len(f)), 1],
                    s=5, label='Train Cluster {}'.format(idx))

        len_train += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_train_list)))
    plt.show()
    plt.close()

    # show each cluster separately on all test data
    len_test = len(X_train)
    for idx, f in enumerate(X_test_list):
        plt.scatter(embeddings[len_test:(len_test + len(f)), 0],
                    embeddings[len_test:(len_test + len(f)), 1],
                    s=5, label='Test Cluster {}'.format(idx))
        len_test += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_test_list)))
    plt.show()
    plt.close()


def plot_KM(y_list, cluster_method, data_name,
            is_train=True, is_lifelines=True,
            seed=42, num_inst=1000, num_feat=200,
            is_expert=False, shape=[], scale=[], t_horizon=10):
    """This function is to plot the Kaplan-Meier curve regarding different clusters"""

    if is_train:
        stage = 'train'
    else:
        stage = 'test'

    group_indicator = []
    for idx, cluster in enumerate(y_list):
        group_indicator.append([idx] * len(cluster))
    group_indicator = np.concatenate(group_indicator)

    if is_lifelines:
        results = multivariate_logrank_test([item[1] for item in np.concatenate(y_list)], # item 1 is the survival time
                                            group_indicator,
                                            [int(item[0]) for item in np.concatenate(y_list)]) # item 0 is the event
        chisq, pval = results.test_statistic, results.p_value
    else:
        chisq, pval = compare_survival(np.concatenate(y_list), group_indicator)

    print('Test statistic of {}: {:.4e}'.format(stage, chisq))
    print('P value of {}: {:.4e}'.format(stage, pval))
    figure(figsize=(8, 6), dpi=80)
    for idx, cluster in enumerate(y_list):  # each element in the y_list is a cluster
        # use lifelines' KM tool to estimate and plot KM
        # this will provide confidence interval
        if len(cluster) == 0:
            continue
        if is_lifelines:
            kmf = KaplanMeierFitter()
            kmf.fit([item[1] for item in cluster], event_observed=[item[0] for item in cluster],
                    label='Cluster {}, #{}'.format(idx, len(cluster)))
            kmf.plot_survival_function(ci_show=False, show_censors=True)
        else:
            # use scikit-survival's KM tool to estimate and plot KM
            # this does not provide confidence interval
            x, y = kaplan_meier_estimator([item[0] for item in cluster], [item[1] for item in cluster])
            plt.step(x, y, where="post", label='Cluster {}, #{}'.format(idx, len(cluster)))
            # plt.ylim(0, 1)

    if is_expert:
        step = 100
        for i in range(len(shape)):
            k = shape[i]
            b = scale[i]
            s = np.zeros(step)
            t_space = np.linspace(0, t_horizon, step)
            for j in range(step):
                s[j] = -(np.power(np.exp(b) * t_space[j], np.exp(k)))
            plt.plot(t_space, s, label='Expert Distribution {}'.format(i))

    plt.title("LogRank: {:.2f}".format(chisq), fontsize=18)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Survival Probability", fontsize=18)
    plt.legend(fontsize=18)

    if data_name == 'sim':
        plt.savefig('./Figures/{}_{}_KM_plot_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, num_inst, num_feat, seed))
    else:
        plt.savefig('./Figures/{}_{}_KM_plot_#clusters{}_{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, seed))
    plt.show()
    plt.close()
    return pval, chisq


