import torch
from spikingjelly.activation_based import neuron,surrogate
from typing import Callable, Optional

class Bernoulli_neuron(neuron.IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
    #here we provide an simple multi-timstep realization of the bernoulli neuron
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        # Stochastic spike using Bernoulli distribution
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            accumulated = torch.sum(x_seq[t], dim=3)
            normalized = accumulated / x_seq.shape[3]
            bernoulli_spike_train = torch.bernoulli(normalized.unsqueeze(-1).expand_as(x_seq[t]))
            spike_seq[t] = bernoulli_spike_train
        return spike_seq, v