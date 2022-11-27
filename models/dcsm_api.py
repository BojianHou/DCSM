"""
This module is a wrapper around torch implementations and
provides a convenient API to train Deep Clustering Survival Machines.
"""

from .dcsm_torch import DeepClusteringSurvivalMachinesTorch
from utils import losses as losses
from utils.model_utils import train_dcsm
from utils.model_utils import _reshape_tensor_with_nans
import torch
import numpy as np


class DCSMBase():
    """Base Class"""

    def __init__(self, k=3, layers=None, distribution="Weibull",
                 temp=1000., discount=1.0, random_state=42, fix=False, is_seed=False):
        self.k = k
        self.layers = layers
        self.dist = distribution
        self.temp = temp
        self.discount = discount
        self.fitted = False

        self.random_state = random_state
        self.fix = fix
        self.is_seed = is_seed

    def _gen_torch_model(self, inputdim, optimizer, risks):
        """Helper function to return a torch model."""

        return DeepClusteringSurvivalMachinesTorch(inputdim,
                                         k=self.k,
                                         layers=self.layers,
                                         dist=self.dist,
                                         temp=self.temp,
                                         discount=self.discount,
                                         optimizer=optimizer,
                                         risks=risks,
                                         random_state=self.random_state,
                                         fix=self.fix,
                                         is_seed=self.is_seed)  # .cuda()

    def fit(self, x, t, e, vsize=0.15, val_data=None,
            iters=10000, learning_rate=1e-3, batch_size=100,
            elbo=True, optimizer="Adam", random_state=100):

        r"""This method is used to train an instance of the DCSM model.

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: np.ndarray
            A numpy array of the event/censoring times, \( t \).
        e: np.ndarray
            A numpy array of the event/censoring indicators, \( \delta \).
            \( \delta = 1 \) means the event took place.
        vsize: float
            Amount of data to set aside as the validation set.
        val_data: tuple
            A tuple of the validation dataset. If passed vsize is ignored.
        iters: int
            The maximum number of training iterations on the training dataset.
        learning_rate: float
            The learning rate for the `Adam` optimizer.
        batch_size: int
            learning is performed on mini-batches of input data. this parameter
            specifies the size of each mini-batch.
        elbo: bool
            Whether to use the Evidence Lower Bound for optimization.
            Default is True.
        optimizer: str
            The choice of the gradient based optimization method. One of
            'Adam', 'RMSProp' or 'SGD'.
        random_state: float
            random seed that determines how the validation set is chosen.

        """
        # to see the results fluctuating among all epochs
        x, x_test = x
        t, t_test = t
        e, e_test = e

        x_test = torch.from_numpy(x_test).double().cuda()
        t_test = torch.from_numpy(t_test).double().cuda()
        e_test = torch.from_numpy(e_test).double().cuda()

        processed_data = self._preprocess_training_data(x, t, e,
                                                        vsize, val_data,
                                                        random_state)
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data


        inputdim = x_train.shape[-1]

        maxrisk = int(np.nanmax(e_train.cpu().numpy()))
        model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk).cuda()
        model, _ = train_dcsm(model,
                             x_train, t_train, e_train,
                             x_test, t_test, e_test,
                             n_iter=iters,
                             lr=learning_rate,
                             elbo=elbo,
                             bs=batch_size)

        self.torch_model = model.eval()
        self.fitted = True

        return self

    def compute_nll(self, x, t, e):
        r"""This function computes the negative log likelihood of the given data.
        In case of competing risks, the negative log likelihoods are summed over
        the different events' type.

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: np.ndarray
            A numpy array of the event/censoring times, \( t \).
        e: np.ndarray
            A numpy array of the event/censoring indicators, \( \delta \).
            \( \delta = r \) means the event r took place.

        Returns:
          float: Negative log likelihood.
        """
        if not self.fitted:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `_eval_nll`.")
        processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
        _, _, _, x_val, t_val, e_val = processed_data
        x_val, t_val, e_val = x_val, \
                              _reshape_tensor_with_nans(t_val), \
                              _reshape_tensor_with_nans(e_val)
        loss = 0
        for r in range(self.torch_model.risks):
            loss += float(losses.conditional_loss(self.torch_model,
                                                  x_val, t_val, e_val, elbo=False,
                                                  risk=str(r + 1)).detach().cpu().numpy())
        return loss

    def _preprocess_test_data(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).double().cuda()
        else:
            return x.cuda()

    def _preprocess_training_data(self, x, t, e, vsize, val_data, random_state):

        idx = list(range(x.shape[0]))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        x_train, t_train, e_train = x[idx], t[idx], e[idx]

        x_train = torch.from_numpy(x_train).double().cuda()
        t_train = torch.from_numpy(t_train).double().cuda()
        e_train = torch.from_numpy(e_train).double().cuda()

        if val_data is None:

            vsize = int(vsize * x_train.shape[0])
            x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

            x_train = x_train[:-vsize]
            t_train = t_train[:-vsize]
            e_train = e_train[:-vsize]

        else:

            x_val, t_val, e_val = val_data

            x_val = torch.from_numpy(x_val).double().cuda()
            t_val = torch.from_numpy(t_val).double().cuda()
            e_val = torch.from_numpy(e_val).double().cuda()

        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def predict_mean(self, x, risk=1):
        r"""Returns the mean Time-to-Event \( t \)

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        Returns:
          np.array: numpy array of the mean time to event.

        """

        if self.fitted:
            x = self._preprocess_test_data(x)
            scores = losses.predict_mean(self.torch_model, x, risk=str(risk))
            return scores
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_mean`.")

    def predict_shape_scale(self, x, risk=1):
        if self.fitted:
            x = self._preprocess_test_data(x)
            shapes, scales, logits = self.torch_model.forward(x, risk=str(risk))
            shape = torch.sum(torch.mul(shapes, logits), dim=1) / torch.sum(logits, dim=1)
            scale = torch.sum(torch.mul(scales, logits), dim=1) / torch.sum(logits, dim=1)
            return shape.detach().cpu().numpy(), scale.detach().cpu().numpy()

    def predict_risk(self, x, t, risk=1):
        r"""Returns the estimated risk of an event occuring before time \( t \)
        \( \widehat{\mathbb{P}}(T\leq t|X) \) for some input data \( x \).

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: list or float
            a list or float of the times at which survival probability is
            to be computed
        Returns:
          np.array: numpy array of the risks at each time in t.

        """

        if self.fitted:
            return 1 - self.predict_survival(x, t, risk=str(risk))
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_risk`.")

    def predict_survival(self, x, t, risk=1):
        r"""Returns the estimated survival probability at time \( t \),
          \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: list or float
            a list or float of the times at which survival probability is
            to be computed
        Returns:
          np.array: numpy array of the survival probabilites at each time in t.

        """
        x = self._preprocess_test_data(x)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            scores = losses.predict_cdf(self.torch_model, x, t, risk=str(risk))
            return np.exp(np.array(scores)).T
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_survival`.")

    def predict_pdf(self, x, t, risk=1):
        r"""Returns the estimated pdf at time \( t \),
          \( \widehat{\mathbb{P}}(T = t|X) \) for some input data \( x \).

        Parameters
        ----------
        x: np.ndarray
            A numpy array of the input features, \( x \).
        t: list or float
            a list or float of the times at which pdf is
            to be computed
        Returns:
          np.array: numpy array of the estimated pdf at each time in t.

        """
        x = self._preprocess_test_data(x)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            scores = losses.predict_pdf(self.torch_model, x, t, risk=str(risk))
            return np.exp(np.array(scores)).T
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_survival`.")

    def predict_phenotype(self, x, risk=1):
        r"""Returns the weights for each input x
  
          Parameters
          ----------
          x: np.ndarray
              A numpy array of the input features, \( x \).
          Returns:
            np.array: numpy array of the weight for each base distribution.

          """
        x = self._preprocess_test_data(x)
        shape, scale, logits = self.torch_model.forward(x, risk=str(risk))
        cluster_tag = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return cluster_tag, shape[0], scale[0]


class DeepClusteringSurvivalMachines(DCSMBase):
    """A Deep Survival Machines model.

      This is the main interface to a Deep Survival Machines model.
      A model is instantiated with approporiate set of hyperparameters and
      fit on numpy arrays consisting of the features, event/censoring times
      and the event/censoring indicators.

      For full details on Deep Survival Machines, refer to our paper [1].

      References
      ----------
      [1] <a href="https://arxiv.org/abs/2003.01176">Deep Survival Machines:
      Fully Parametric Survival Regression and
      Representation Learning for Censored Data with Competing Risks."
      arXiv preprint arXiv:2003.01176 (2020)</a>

      Parameters
      ----------
      k: int
          The number of underlying parametric distributions.
      layers: list
          A list of integers consisting of the number of neurons in each
          hidden layer.
      distribution: str
          Choice of the underlying survival distributions.
          One of 'Weibull', 'LogNormal'.
          Default is 'Weibull'.
      temp: float
          The logits for the gate are rescaled with this value.
          Default is 1000.
      discount: float
          a float in [0,1] that determines how to discount the tail bias
          from the uncensored instances.
          Default is 1.

      Example
      -------
      from dcsm import DeepSurvivalMachines
      model = DeepSurvivalMachines()
      model.fit(x, t, e)

      """

    def __call__(self):
        if self.fitted:
            print("A fitted instance of the Deep Survival Machines model")
        else:
            print("An unfitted instance of the Deep Survival Machines model")

        print("Number of underlying distributions (k):", self.k)
        print("Hidden Layers:", self.layers)
        print("Distribution Choice:", self.dist)


