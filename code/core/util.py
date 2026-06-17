import torch
import datetime
import time

def check_available_device():
    """
    使用可能なデバイスをtorch.device()で返します
    """
    if torch.cuda.is_available(): # NVIDIA GPU
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available(): # MacOS(AppleSilicon)
        device = torch.device('mps')
    else: # CPU
        device = torch.device('cpu')
    return device

def get_time_info():
    """
    実行時のunix時間と'YYYY/MM/DD hh:mm:ss'を返します
    """
    ut = time.time() # unixtime
    dt = datetime.datetime.fromtimestamp(ut).strftime('%Y/%m/%d %H:%M:%S') # datetimeに変換
    return ut, dt

