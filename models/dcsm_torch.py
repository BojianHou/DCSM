"""Torch model definitons for the Deep Clustering Survival Machines model

This includes definitons for the Deep Clustering Survival Machines module.
The main interface is the DeepClusteringSurvivalMachines class which inherits
from torch.nn.Module.

"""

import torch.nn as nn
import torch
import numpy as np


def create_representation(inputdim, layers, activation):
    r"""Helper function to generate the representation function for DCSM.

  Deep Clustering Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Clustering Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  layers: list
      A list consisting of the number of neurons in each hidden layer.
  activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

  Returns
  ----------
  an MLP with torch.nn.Module with the specfied structure.

  """

    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=False))  # .cuda()
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class DeepClusteringSurvivalMachinesTorch(nn.Module):
    """A Torch implementation of Deep Clustering Survival Machines model.

  This is an implementation of Deep Clustering Survival Machines model in torch.
  It inherits from the torch.nn.Module class and includes references to the
  representation learning MLP, the parameters of the underlying distributions
  and the forward function which is called whenver data is passed to the
  module. Each of the parameters belongs to nn.Parameters and torch automatically
  keeps track and computes gradients for them.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  activation: str
      Choice of activation function for the MLP representation.
      One of 'ReLU6', 'ReLU' or 'SeLU'.
      Default is 'ReLU6'.
  dist: str
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

  """

    def _init_dcsm_layers(self, lastdim):

        if self.is_seed:  # if is_seed is true, means we use the random seed to fix the initialization
            print('random seed for torch model initialization is: ', self.random_state)
            torch.manual_seed(self.random_state)  # fix the initialization
        if self.dist in ['Weibull']:
            self.act = nn.SELU()
            if self.fix:  # means using fixed base distribution
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()
            else:
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
        else:
            raise NotImplementedError('Distribution: ' + self.dist + ' not implemented' +
                                      ' yet.')

        self.gate = nn.ModuleDict({str(r + 1): nn.Sequential(
            nn.Linear(lastdim, self.k, bias=False)
        ) for r in range(self.risks)})  # .cuda()

        if self.fix == False:  # means using varied base distribution by discarding these parameters
            self.scaleg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

            self.shapeg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

    def __init__(self, inputdim, k, layers=None, dist='Weibull',
                 temp=1000., discount=1.0, optimizer='Adam',
                 risks=1, random_state=42, fix=False, is_seed=False):
        super(DeepClusteringSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.dist = dist
        self.temp = float(temp)
        self.discount = float(discount)
        self.optimizer = optimizer
        self.risks = risks

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self.random_state = random_state
        self.fix = fix
        self.is_seed = is_seed

        self._init_dcsm_layers(lastdim)
        self.embedding = create_representation(inputdim, layers, 'ReLU6')

    def forward(self, x, risk='1'):
        """The forward function that is called when data is passed through DCSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
        xrep = self.embedding(x)
        dim = x.shape[0]

        if self.fix:  # means using fixed base distributions
            return (self.shape[risk].expand(dim, -1).cuda(),
                    self.scale[risk].expand(dim, -1).cuda(),
                    self.gate[risk](xrep) / self.temp)
        else:
            return (self.act(self.shapeg[risk](xrep)) + self.shape[risk].expand(dim, -1),
                    self.act(self.scaleg[risk](xrep)) + self.scale[risk].expand(dim, -1),
                    self.gate[risk](xrep) / self.temp)

    def get_shape_scale(self, risk='1'):
        return self.shape[risk], self.scale[risk]


def create_conv_representation(inputdim, hidden,
                               typ='ConvNet', add_linear=True):
    r"""Helper function to generate the representation function for DCSM.

  Deep Clustering Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Convolutional Neural
  Network (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Clustering Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: tuple
      Dimensionality of the input image.
  hidden: int
      The number of neurons in each hidden layer.
  typ: str
      Choice of convolutional neural network: One of 'ConvNet'

  Returns
  ----------
  an ConvNet with torch.nn.Module with the specfied structure.

  """

    if typ == 'ConvNet':
        embedding = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU6(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU6(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.ReLU6(),
        )

    if add_linear:
        dummyx = torch.ones((10, 1) + inputdim)
        dummyout = embedding.forward(dummyx)
        outshape = dummyout.shape

        embedding.add_module('linear', torch.nn.Linear(outshape[-1], hidden))
        embedding.add_module('act', torch.nn.ReLU6())

    return embedding