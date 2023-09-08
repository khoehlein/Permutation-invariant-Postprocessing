from torch import Tensor, nn
import model.custom_activations as ca

def to_device_if_not(data: Tensor, device):
    if device is None or data.device == device:
        return data
    return data.to(device)


def get_activation(activation,activation_kws=None):
    if activation_kws is None:
        activation_kws = {}
    act_class = None
    try:
        act_class = getattr(nn, activation)
    except AttributeError:
        pass
    if act_class is None:
        try:
            act_class = getattr(ca, activation)
        except AttributeError:
            pass
    assert act_class is not None
    return act_class(**activation_kws)
