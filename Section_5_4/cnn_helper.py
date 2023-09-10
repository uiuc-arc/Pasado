import torch
from torch import nn

from model import Normalization, Conv

import sys

sys.path.insert(0, '../forward_mode_tensorized_src')
from SimpleZono import *
from Duals import *
from precise_transformer import *


def NormalizationInterval(layer: Normalization, x: DualIntervalTensor):
    return (x - layer.mean.flatten()) / layer.sigma.flatten()


def NormalizationZonotope(layer: Normalization, x: DualZonotope):
    return (x - layer.mean.flatten()) / layer.sigma.flatten()


def ConvInterval(layer: Conv, x: DualIntervalTensor, init_shape: (int, int, int)):
    if len(init_shape) != 3:
        raise ValueError(
            "The `init_shape` argument must a tuple consisting of (n_channels, input_width, input_height)!")

    shape_ = init_shape
    for layer_ in layer.layers.children():
        if isinstance(layer_, Normalization):
            x = NormalizationInterval(layer_, x)
        elif isinstance(layer_, nn.Sigmoid):
            x = Sigmoid_di(x)
        elif isinstance(layer_, nn.Linear):
            x = Affine_di(x, layer_.weight.T) + layer_.bias
        elif isinstance(layer_, nn.Softplus):
            x = SmoothRelu_di(x)
        elif isinstance(layer_, nn.Conv2d):
            x, shape_ = Conv2D_di(x, layer_, shape_)
        elif isinstance(layer_, (nn.Flatten, nn.Dropout)):
            pass  # No-op.
        else:
            raise NotImplementedError(f"Module {type(layer_)} not implemented!")

    return x


def ConvZonotope(layer: Conv, x: DualZonotope, init_shape: (int, int, int)):
    if len(init_shape) != 3:
        raise ValueError(
            "The `init_shape` argument must a tuple consisting of (n_channels, input_width, input_height)!")

    shape_ = init_shape
    for layer_ in layer.layers.children():
        if isinstance(layer_, Normalization):
            x = NormalizationZonotope(layer_, x)
        elif isinstance(layer_, nn.Sigmoid):
            x = SigmoidDualZonotope(x)
        elif isinstance(layer_, nn.Linear):
            x = AffineDualZonotope(x, layer_.weight.T) + layer_.bias
        elif isinstance(layer_, nn.Softplus):
            x = SmoothReluDualZonotope(x)
        elif isinstance(layer_, (nn.Flatten, nn.Dropout)):
            pass  # No-op.
        elif isinstance(layer_, nn.Conv2d):
            x, shape_ = Conv2dDualZonotope(x, layer_, shape_)
        else:
            raise NotImplementedError(f"Module {type(layer_)} not implemented!")

    return x


def ConvPasado(layer: Conv, x: DualZonotope, init_shape: (int, int, int)):
    if len(init_shape) != 3:
        raise ValueError(
            "The `init_shape` argument must a tuple consisting of (n_channels, input_width, input_height)!")

    shape_ = init_shape
    for layer_ in layer.layers.children():
        if isinstance(layer_, Normalization):
            x = NormalizationZonotope(layer_, x)
        elif isinstance(layer_, nn.Sigmoid):
            x = PreciseSigmoidDualZonotope(x)
        elif isinstance(layer_, nn.Softplus):
            x = PreciseSoftplusDualZonotope(x)
        elif isinstance(layer_, nn.Linear):
            x = AffineDualZonotope(x, layer_.weight.T) + layer_.bias
        elif isinstance(layer_, (nn.Flatten, nn.Dropout)):
            pass  # No-op.
        elif isinstance(layer_, nn.Conv2d):
            x, shape_ = Conv2dDualZonotope(x, layer_, shape_)
        else:
            raise NotImplementedError(f"Module {type(layer_)} not implemented!")

    return x
