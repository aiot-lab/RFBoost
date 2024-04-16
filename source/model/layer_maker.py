import torch.nn as nn
from .norm import GlobalLayerNorm, CumulativeLayerNorm1d

def choose_nonlinear(name, **kwargs):
    if name == 'relu':
        nonlinear = nn.ReLU()
    elif name == 'sigmoid':
        nonlinear = nn.Sigmoid()
    elif name == 'softmax':
        assert 'dim' in kwargs, "dim is expected for softmax."
        nonlinear = nn.Softmax(**kwargs)
    elif name == 'tanh':
        nonlinear = nn.Tanh()
    elif name == 'leaky-relu':
        nonlinear = nn.LeakyReLU()
    else:
        raise NotImplementedError("Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(name))
    
    return nonlinear

def choose_rnn(name, **kwargs):
    if name == 'rnn':
        rnn = nn.RNN(**kwargs)
    elif name == 'lstm':
        rnn = nn.LSTM(**kwargs)
    elif name == 'gru':
        rnn = nn.GRU(**kwargs)
    else:
        raise NotImplementedError("Invalid RNN is specified. Choose 'rnn', 'lstm', or 'gru' instead of {}.".format(name))
    
    return rnn

def choose_layer_norm(name, num_features, causal=False, eps=1e-12, **kwargs):
    if name == 'cLN':
        layer_norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif name == 'gLN':
        if causal:
            raise ValueError("Global Layer Normalization is NOT causal.")
        layer_norm = GlobalLayerNorm(num_features, eps=eps)
    elif name in ['BN', 'batch', 'batch_norm']:
        n_dims = kwargs.get('n_dims') or 1
        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError("n_dims is expected 1 or 2, but give {}.".format(n_dims))
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm