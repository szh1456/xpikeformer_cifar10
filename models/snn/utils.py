import torch
from torch import nn
from spikingjelly.activation_based import base
from spikingjelly.activation_based.encoding import PoissonEncoder


def spike_coding(input, n_timesteps, is_sto):
    b, c, h, w = input.shape
    out_spike = torch.full((n_timesteps,b, c, h, w),0,dtype=torch.float32)
    if is_sto:
        pe = PoissonEncoder()
        for t in range(n_timesteps):
            out_spike[t] = pe(input)
        return out_spike
    else:
        return input.unsqueeze(0).expand(n_timesteps, b, c, h, w) 

def reset(network: torch.nn.Module):
    reset_net(network)

def reset_net(net: nn.Module):
    for m in net.modules():
        if isinstance(m, base.MemoryModule) and hasattr(m, 'reset'):
            m.reset()