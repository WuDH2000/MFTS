import torch

max_filter = torch.nn.MaxPool1d(kernel_size=256, stride=128, padding=128)
# max_filter = torch.nn.MaxPool1d(kernel_size=256, stride=128)
max_filter.requires_grad_ = False
def get_env_points(wav):
    res =  max_filter(torch.abs(wav))
    return res

def get_env_by_inpterp(points, length = 48000):
    return torch.nn.functional.interpolate(points, (length), mode = 'linear')

def get_env(wav):
    res = max_filter(torch.abs(wav))
    return torch.nn.functional.interpolate(res, (wav.shape[-1]), mode = 'linear')