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


# 导入 Accelerate 库的核心组件，用于分布式训练和混合精度训练
from accelerate import Accelerator, InitProcessGroupKwargs
# 导入 Accelerate 工具函数，用于跨进程数据填充和广播
from accelerate.utils import pad_across_processes, broadcast
# 导入 defaultdict，用于创建具有默认值的字典
from collections import defaultdict
# 导入数据类相关装饰器和函数，用于定义配置类和数据结构
from dataclasses import dataclass, field, asdict
# 导入 Datasets 库的核心组件，用于数据集加载、处理和操作
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
# 导入时间间隔类，用于设置训练超时和时间管理
from datetime import timedelta
# 导入 functools 模块的 partial 函数，用于创建偏函数（固定部分参数的函数）
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code
from src.utils import set_seed, floatify, compute_ETA
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW, get_constant_schedule_with_warmup
import wandb
import pandas as pd
import shutil

# 使用 partial 函数创建一个预配置的 tqdm 进度条函数
# partial 的作用是固定 tqdm 的部分参数：ncols=0（自动调整宽度）和 leave=False（完成后不保留进度条）
# 这样后续调用 tqdm() 时会自动应用这些默认参数，避免重复设置
tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10
instruction=None
cot_trigger=None
answer_trigger=None
def setup_cot(src_name):
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric']
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '
    return 


post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: float(x.replace('"','').replace("'",'').strip()),
    'mathqa-numeric': lambda x: float(x.replace('"','').replace("'",'').strip()),
}



### the answer_cot is a list of answer_cot

post_process_final_answer_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],


}





def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'],'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'],'r'))),
        })






def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(
        save_path, 
        is_main_process = accelerator.is_main_process, 
        save_function=accelerator.save, 
        state_dict=accelerator.get_state_dict(model),
    )

    tokenizer.save_pretrained(save_path)

    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths)> args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            shutil.rmtree(ckpt_to_be_removed)











def evaluatte_generation(
    args,
    model,
    dataset,
    dataloader,
    tokenizer,
):
    pass