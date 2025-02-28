import torch
from torch import Tensor

def torch_conv_batch(in1, in2):
    b, s, t = in1.shape
    o = torch.conv1d(torch.flip(in1, (0, 2)), in2, padding=in1.shape[2], groups=s)
    o = o[:, :, in1.shape[2]+1:]
    o = torch.flip(o, (0, 2))
    return o

def TAC_2TC_KM(idif, t, k_params, step=0.1): #optimised for batch processing
    k1, k2, k3, Vb = k_params.T.unbind(0)
    k1 = k1.unsqueeze(1).unsqueeze(2)# * 2
    k2 = k2.unsqueeze(1).unsqueeze(2)# * 3
    k3 = k3.unsqueeze(1).unsqueeze(2)
    Vb = Vb.unsqueeze(1)
    t = t.repeat(k1.shape[0], 1, 1)
    a = idif.unsqueeze(0)
    e = (k2+k3) * t
    b = k1 / (k2+k3) * (k3 + k2*torch.exp(-e)) # assuming no gluconeogenesis
    c = torch_conv_batch(a, b) * step
    TAC = ((1-Vb) * (c.squeeze(0))) + (Vb * (a.squeeze(0)))
    TAC.requires_grad_()
    return TAC

def torch_interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indexes = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indexes = torch.clamp(indexes, 0, len(m) - 1)

    return m[indexes] * x + b[indexes]