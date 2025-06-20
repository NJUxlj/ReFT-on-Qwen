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
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Gen Loop'):
        output_ = model.module.generate(
                        **batch['generate_prefix_kwargs'], 
                        max_length=args['max_input_length'],
                        output_scores=True,
                        return_dict_in_generate=True,
                        num_beams=1,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
        # output_ 的类型是 transformers.generation.utils.GenerateDecoderOnlyOutput（或类似的 generate 返回对象，取决于 transformers 版本）。
        # 其中 output_.sequences 是一个 torch.LongTensor，形状为 (batch_size, sequence_length)，
        # 表示每个样本生成的 token id 序列。
        # output_ 还包含 scores、sequences_scores 等其他生成相关信息。
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']  # shape = (batch_size, sequence_length)
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True) # 对齐长度
        labels[labels==-100]=tokenizer.pad_token_id  # 这个时候已经不用计算 cross entropy 了， 因此直接把 -100 转为 pad_token 即可， 

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        predictions.extend(preds)
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in labels]
        targets.extend(target)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]


    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = []
        src_name = dataset[0]['item_id'].split("_")[0]
        for pred, tar, item in zip(predictions, targets, dataset):
            pass






def main(args):
    set_seed(args['seed'] + accelerator.process_index)
    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)

    
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    accelerator.print(f'[Vocab size]: {len(tokenizer)}')
    # 如果 tokenizer 的词表大小和模型 embedding 层的大小一致，其实不需要再调用 resize_token_embeddings
    # 只有在自定义 tokenizer 或扩充词表时才需要 resize
    model.resize_token_embeddings(len(tokenizer))

    if accelerator.is_main_process and args['wandb_log']:
        wandb.run.summary.update({
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'vocab_size': len(tokenizer)
        })

    n_epochs = args['n_epochs']
    # 这里原来的计算方式是错的，应该是：总步数 = 总样本数 // (总 batch size) * 轮数
        # 总 batch size = 单卡 batch size × 进程数 × 梯度累积步数
    # num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs) // args['gradient_accumulation_steps']
    total_batch_size = args['batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']
    num_training_steps = (len(train_dataset) * n_epochs) // total_batch_size
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)


    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        #  总训练 batch size（考虑并行、分布式和梯度累积）= 单卡 batch size × 进程数 × 梯度累积步数
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size']*accelerator.num_processes*args['gradient_accumulation_steps']}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )   









if __name__ == "__main__":

    from transformers import HfArgumentParser
    NONE_INT = -100 
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str 
        test_file: str
        batch_size: int = field(default=4)
        eval_batch_size: int = field(default=8)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        gradient_accumulation_steps: int = field(default=1)
        keep_num_ckpt: int = field(default=1)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='python')

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None

    accelerator = Accelerator(InitProcessGroupKwargs={})
    accelerator.print(args)
    accelerator.print(json.dumps(args, ensure_ascii=False, indent=4))

    main(args)