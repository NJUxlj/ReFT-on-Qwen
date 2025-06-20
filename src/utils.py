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



from trl.core import masked_mean, masked_var
def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """

    allgather_values = allgather(values)  # shape = (world_size, bs, seqlen)

    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask  =allgather(mask)  # shape = (world_size, bs, seqlen)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)

    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8) # 作用， 将 values 白化， 使得白化后的 values 的均值为 0， 方差为 1， 将 values 的范围限制在 [-1, 1] 之间

    if shift_mean:
        whitened += global_mean
    return whitened  # shape = (bs, seqlen)


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount):
    """
    计算折扣累计和（discounted cumulative sum）。

    该函数常用于强化学习中，用于计算奖励序列的折扣累计和（如 GAE、回报等）。
    给定奖励序列 rewards = [r0, r1, r2, ..., rN] 和折扣因子 discount，
    输出为 [r0 + discount*r1 + discount^2*r2 + ...,
           r1 + discount*r2 + ...,
           ...,
           rN]

    工作流程如下：
    1. rewards[::-1]：将奖励序列反转，便于从后往前递推累计和。
    2. scipy_signal.lfilter([1], [1, -discount], x)：使用线性滤波器高效实现递推公式。
       - 该滤波器实现了 y[n] = x[n] + discount * y[n-1]，即 y[n] = rewards[n] + discount * y[n+1] 的反向递推。
    3. 最后再[::-1]反转回来，得到正序的折扣累计和。


    # 举例：
    
    - 原始奖励为： 
    [r1, r2, r3,..., r_{N-1}, r_N]

    - 反转后的奖励为：
    [r_N, r_{N-1}, ..., r_3, r_2, r_1]

    - 使用 lfilter 计算折扣累计和：
    R_N = r_N + discount * 0     # 注，第 N+1 步的奖励为 0， 这应该已经包含在 reward 矩阵中了， 
                                    # reward 矩阵初始时为全 0，  初始时的 reward.shape = (bs, seqlen)
                                    # 但由于最后一步的奖励无意义，我们 只填充 reward[:, :-1]

    R_{N-1} = r_{N-1} + discount * R_N

    R_{N-2} = r_{N-2} + discount * R_{N-1}

    ...

    R_1 = r_1 + discount * R_2 = r1 + discount * (r2 + discount * (r3 + discount * ...))

    - 最后再反转回来，得到正序的折扣累计和：
    [R_1, R_2, ..., R_{N-1}, R_N]


    Args:
        rewards (np.ndarray or list): 奖励序列
        discount (float): 折扣因子

    Returns:
        np.ndarray: 折扣累计和序列
    """
    # 先将奖励序列反转，便于从后往前递推
    # 使用lfilter高效计算折扣累计和
    # 最后再反转回来，得到正序结果
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]




from datetime import timedelta
def compute_ETA(tqdm_t, num_period=1):
    rate = tqdm_t.format_dict["rate"]

    time_per_period = tqdm_t.total / rate if rate and tqdm_t.total else 0   # * Seconds *
    period_remaining = (tqdm_t.total - tqdm_t.n)/ rate if rate and tqdm_t.total else 0

    remaining = time_per_period * (num_period-1) + period_remaining

    return timedelta(seconds = remaining)

    