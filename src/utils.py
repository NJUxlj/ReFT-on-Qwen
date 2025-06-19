# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import random
import numpy as np
import torch
import signal
import json 




class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_numeric(value)->bool:
    try:
        value = float(value)
        return True
    except Exception as e:
        return False

def floatify(s):
    try:
        return float(s)
    except:
        return None
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_data(file:str, data)->None:
    with open() as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


from torch.distributed import all_reduce, ReduceOp
def do_gather(var):
    var = torch.FloatTensor(var).cuda()
    all_reduce(var, op = ReduceOp.SUM)
    var = var.cpu().numpy().tolist()
    return var


def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.


    该函数是对 torch.distributed.all_gather 的一个语法糖封装，用于在分布式训练中收集所有进程上的 tensor。
        输入 tensor 形状为 (bs, ...)，返回的 allgather_tensor 形状为 (world_size, bs, ...)
        这样可以方便地在主进程或所有进程上拿到所有进程的数据，常用于分布式训练中的统计、同步等场景。

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """

    # 该函数是对 torch.distributed.all_gather 的一个语法糖封装，用于在分布式训练中收集所有进程上的 tensor。
    # 输入 tensor 形状为 (bs, ...)，返回的 allgather_tensor 形状为 (world_size, bs, ...)
    # 这样可以方便地在主进程或所有进程上拿到所有进程的数据，常用于分布式训练中的统计、同步等场景。

    if group is None:
        # 如果没有指定 group，则使用默认的全局通信组
        group = torch.distributed.group.WORLD
    
    # 创建一个和 world_size 一样长的列表，每个元素 shape 和 tensor 一样，初始为 0
        # torch.zero_like(tensor) 用于生成和 tensor 形状、类型一致的全 0 tensor
    allgather_tensor  = [torch.zero_like(tensor) for _ in range(group.size())]
    # 调用 all_gather，把当前进程的 tensor 收集到 allgather_tensor 列表中
    torch.distributed.all_gather(allgather_tensor, tensor, group = group)

    # 把收集到的 tensor 列表堆叠成一个新的 tensor，shape 变为 (world_size, bs, ...)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)

    return allgather_tensor





from datetime import timedelta
def compute_ETA(tqdm_t, num_period=1):
    rate = tqdm_t.format_dict["rate"]

    time_per_period = tqdm_t.total / rate if rate and tqdm_t.total else 0   # * Seconds *
    period_remaining = (tqdm_t.total - tqdm_t.n)/ rate if rate and tqdm_t.total else 0

    remaining = time_per_period * (num_period-1) + period_remaining

    return timedelta(seconds = remaining)

    