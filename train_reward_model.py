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
from accelerate import Accelerator, InitProcessGroupKwargs
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import numpy as np
import os
from src.utils import set_seed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
import wandb
import shutil
# partial函数对tqdm进行包装，默认设置进度条的ncols为0（自动适应终端宽度），leave为False（迭代结束后不保留进度条）。
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




def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'],'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'],'r'))),
        })
        accelerator.print('Raw data:', raw_dataset)
        src_name = raw_dataset['train'][0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        setup_cot(src_name)
        accelerator.print('Using instruction:', instruction)
        accelerator.print('Using cot_trigger:', cot_trigger)
        accelerator.print('Using answer_trigger:', answer_trigger)

        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())

            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k:item_values[i] for i, k in enumerate(all_keys)}  
                item_id, question, answer_value, predictions, vote_correctness = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item['predictions'], \
                        item['is_correct']   # 经过 majority voting 判断后，是否正确
                
                question, answer_value = question.strip(), answer_value.strip()
                unique_ = set()

                for sample in predictions:
                    prediction_cot, prediction_correctness, prediction_value = sample['completion'], sample['correctness'], sample['solving_res']
                    # deduplication
                    if prediction_cot in unique_:
                        continue
                    unique_.add(prediction_cot)

                    input = f'{instruction}{question}{cot_trigger}{prediction_cot}'
                    input_encode = tokenizer(input, add_special_tokens=False)
                    input_ids = input_encode['input_ids']
                    attention_mask = [1] *  len(input_ids)
                    labels = prediction_correctness # 奖励分数



                    # Truncation and Filtering
                    input_ids = input_ids[args['max_input_length']:]
                    attention_mask = attention_mask[args['max_input_length']:]
                    assert tokenizer.pad_token_id not in input_ids, input_ids

                    ##
                    new_batch['input_ids'].append(input_ids)
                    new_batch['labels'].append(labels)
                    new_batch['attention_mask'].append(attention_mask)
                    ##
                    new_batch['item_id'].append(item_id)
                    new_batch['question'].append(question)
                    new_batch['prediction_cot'].append(prediction_cot)
                    new_batch['prediction_correctness'].append(prediction_correctness)
                    new_batch['prediction_value'].append(prediction_value)
                    new_batch['answer_value'].append(answer_value)
                    new_batch['vote_correctness'].append(vote_correctness)
                
            return new_batch
        
        # load_from_cache_file 参数用于指定在调用 map 函数时，是否使用之前缓存的处理结果。
            # 如果为 True，则会尝试加载之前相同参数下已经缓存的 map 结果，加速数据处理；
            # 如果为 False，则每次都会重新运行 tokenize_fn 进行数据处理，不使用缓存。
        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn,
                fn_kwargs = {'args': args, 'tokenizer': tokenizer},
                batched = True,
                remove_columns = dataset.column_names,
                num_procs = 16,
                load_from_cache_file = True  # 设置为 True 表示优先使用缓存，加快数据处理速度
            ) for mode, dataset in raw_dataset.items()
        })
        accelerator.print('Processed data:', tokenized_dataset)


    def collate_fn(batch, args, tokenizer):
        '''
        补齐以后转为 Dict[Tensor]
        '''
        max_input_length = max([len(item['input_ids']) for item in batch])

        input_ids = []

        attention_mask = []

        labels = []

        for item in batch:
            pass







def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)




def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, test_dataset, test_dataloader, 
                    prefix, epoch, best_eval_log_dict, most_recent_ckpts_paths):
    max_epoch = args['n_epochs']
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)


    model.train()
    epoch_result_dict = defaultdict(list)

    for idx, batch in tqdm(
                            enumerate(train_dataloader),
                            total = len(train_dataloader),
                            disable=not accelerator.is_main_process,
                            desc='Train Loop'):
        
        output = model(**batch['forward_kwargs'])
        # Get some metrics
        loss = output[0]   # shape = (bs, )
        result_dict, extra = {}, None


        accelerator.backward(loss)

        if clip_grad_norm is not None:
            accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        scheduler.step()
        model.zero_grad()


        global_step+=1

        # Step update metric




        # Step evaluating
        eval_log_dict = {}




        # Step logging
        train_log_dict = {}




        # Step saving
        if saving_step_freq is not None and global_step % saving_step_freq == 0:

        







def evaluate_rerank():
    model.eval()
    epoch_result_dict = defaultdict(list)
    predictions = []
    probabilities = []
    targets = []





