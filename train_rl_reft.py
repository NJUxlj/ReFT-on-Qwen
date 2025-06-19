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
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import numpy as np
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
    'gsm8k': lambda x: x.float(x.replace(',', '').strip()),
}


### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs =answer_cot, TIMEOUT=TIMEOUT)],
}


compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}


def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    import deepspeed

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin

    config_kwargs = deepspeed_plugin.deepspeed_config

    if model is not None:
        if hasattr(model, 'config'):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )

            if hidden_size is not None and config_kwargs['zero_optimization']['stage'] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update({
                    # reduce_bucket_size: 控制梯度归约时的桶大小，设置为 hidden_size^2 可以优化通信效率
                    # 较大的桶可以减少通信次数但增加内存使用，这里基于隐藏层大小进行调优
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    # stage3_param_persistence_threshold: 参数持久化阈值，控制哪些参数保持在GPU内存中
                    # 设置为 10 * hidden_size，小于此阈值的参数会持久化在GPU上以减少通信开销
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    # stage3_prefetch_bucket_size: 预取桶大小，控制参数预取的批次大小
                    # 设置为 0.9 * hidden_size^2，在内存使用和预取效率之间取得平衡
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                })

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs['zero_optimization']['stage'] != 3:
        config_kwargs['zero_optimization']['stage'] = 0
    

    # *_ 表示忽略除第一个返回值外的所有其他返回值
    # deepspeed.initialize() 返回多个值：(model, optimizer, dataloader, lr_scheduler)
    # 这里只需要 model，其他返回值用 *_ 忽略掉
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)

    model.eval()
    return model







def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        # make raw dataset
        raw_dataset = Dataset({
            'train': Dataset.from_list(args['train_file']),
            'test': Dataset.from_list(args['test_file'])
        })

        accelerator.print('Raw data:', raw_dataset)

        # make cot related info
        src_name = raw_dataset['train']['item_id'][0].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...

        setup_cot(src_name)


        accelerator.print('Using instruction:', instruction)
        accelerator.print('Using cot_trigger:', cot_trigger)
        accelerator.print('Using answer_trigger:', answer_trigger)


        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)

            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            # 遍历批次中的每个数据项
            # zip(*(batch[k] for k in all_keys)) 的语法解析：
            # 1. (batch[k] for k in all_keys) 是一个生成器表达式，为每个键生成对应的值列表
            # 2. * 操作符将生成器解包，相当于 zip(batch[key1], batch[key2], ...)
            # 3. zip() 函数将多个列表按位置组合，返回元组序列
            # 例如：batch = {'a': [1,2], 'b': [3,4]} -> zip([1,2], [3,4]) -> [(1,3), (2,4)]
            for item_values in zip(*(batch[k] for k in all_keys)):

                # 例如：all_keys=['a','b'], item_values=(1,3) -> {'a':1, 'b':3}
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \


                question = question.strip()
                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot:
                    answer_cot = answer_cot.strip()
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'


                input = f'{instruction}{question}{cot_trigger}'
                output = f'{answer_cot}'
                prefix_text = f'{instruction}{question}{cot_trigger}'



                # Modify for particular datasets and engine
                if src_name in  ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and args['engine'] == 'python':
                    prefix_text += f'def solution():\n    """{question}"""\n'

                input_encode = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)

                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)


                # 这里的 labels 设计是为了实现只对输出部分（即模型生成的答案部分）计算 loss，
                    # 而对输入部分（prompt/question）不计算 loss。
                    # 具体做法是：输入部分的 label 用 -100 填充（PyTorch 的 CrossEntropyLoss 会忽略 -100），
                    # 输出部分的 label 用实际的 token id，最后加上 eos token。
                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)   # 到这里为止还不用 padding
                prefix = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']


                # Truncation
                input_ids = input_ids[:args['max_input_length']]
                labels = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]


                ##
                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                ##
                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['prefix_text'].append(prefix_text)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)

            return new_batch


        

        tokenized_dataset = DatasetDict({
            mode : dataset.map(

            ) for mode, dataset in raw_dataset.items()
        })

        if accelerator.is_main_process and args['wandb_log']:
            wandb.config.update({
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
                'raw_dataset': str(raw_dataset),
                'tokenized_dataset': str(tokenized_dataset),
            })

    

    def collate_fn(batch, args, tokenizer):
        max_input_length = max([len(item['input_ids']) for item in batch])
        max_target_length = max([len(item['labels']) for item in batch])
        max_prefix_length = max([len(item['prefix']) for item in batch])


        input_ids, input_ids_left_padded = [], []
        attention_mask, attention_mask_left_padded = [], []
        labels, labels_left_padded = [], []
        prefix, prefix_left_padded = [], []
        prefix_attention_mask, prefix_attention_mask_left_padded = [], []

        for item in batch:
            pass


        ppo_forward_kwargs = {



        }


        generate_prefix_kwargs = {


        }


        return {
            'ppo_forward_kwargs': ppo_forward_kwargs,
            'generate_prefix_kwargs': generate_prefix_kwargs,
        }




    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'],
                                    num_workers=args['num_workers'], pin_memory=True,
                                    collate_fn = partial(collate_fn, args = args, tokenizer = tokenizer))



    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'],
                                    num_workers=args['num_workers'], pin_memory=True,
                                    collate_fn = partial(collate_fn, args = args, tokenizer = tokenizer))


    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)









def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)

    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            shutil.rmtree(ckpt_to_be_removed)



def rollout(args, model, ref_model, tokenizer, query_tensors, query_tensors_attention_mask, answer_values, src_name):
    model.eval()
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=0.0, top_p=1.0,
            do_sample=True,
            # output_scores=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args['max_gen_length'],
        )

        completed_tensors = gen_output
        # 使用 pad_across_processes 函数对 completed_tensors 进行跨进程填充
            # dim=1: 在第1维（序列长度维度）进行填充
            # pad_index=tokenizer.pad_token_id: 使用分词器的填充token ID作为填充值
            # pad_first=False: 在序列末尾进行填充，而不是在开头填充
            # 这样做是为了确保在分布式训练中，不同进程生成的序列长度保持一致
        completed_tensors = pad_across_processes(completed_tensors, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)

    # Evaluate score
    completed_texts = tokenizer.batch_decode(
        completed_tensors.cpu().numpy().to_list(),
        skip_special_tokens = True,
    )
    programs = [text.strip().split(cot_trigger)[-1].strip() text for text in completed_texts]
    execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
    correctness = []  # 用来存放一个 batch 的奖励分数

    for i, extracted_ans in enumerate(execute_fn(programs)):
        target_value = None

        if extracted_ans is not None:
            if args['engine'] == 'game24' or args['engine'] == 'calcn':
                is_correct = extracted_ans
            else:
                if compare_answer_fn_mapper[src_name](extracted_ans, target_value):
                    is_correct=1
                else:
                    is_correct = 0.1
                    # for mathqa, even though it can executed, if the results is not within a,b,c,d,xxx, still zero reward
                    # because we want to give some reward for the prediction that able to select one of the answer
                    # for example, the executed answer is "{}" in mathqa.
                    # THIS PART IS TO BE DECIDED.
                    # if src_name == 'mathqa' and not (len(extracted_ans) == 1 and extracted_ans.isalpha()):
                    #     is_correct = 0
        else:
            is_correct = 0
        
        correctness.append(is_correct)


    

    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)

    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _dummy2, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        old_logprob = logprobs_from_logits(lm_logits[:, -1, :], labels = model_input_ids[:,1:])  # shape = (bs, seqlen-1)


        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _dummy2, _dummy3 = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, -1, :], labels=model_input_ids[:,1:])   # shape = (bs, seqlen-1)

    # Masking the last prompt token up untils the token before eos_token_id
    prompt_len = query_tensors.size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)

    # Make the kl reward and the full reward
    # ==========================
    # 下面详细注释每一行代码的作用和原理
        # prompt 由 前缀 + <eos> 组成，我们需要确保把 <eos> 和前缀都 mask 掉
        # 这里 mask 的作用是：在 PPO 算法中，我们只对模型生成的答案部分（即 prompt 之后的 token）计算奖励和损失，
        # 而对 prompt 部分（包括 <eos>）不计算奖励和损失，因此需要将这些 token mask 掉。
        # query_tensors.size(1) 表示 prompt 的长度（包括 <eos>），
    # mask[:, query_tensors.size(1)-1:-1] = 1 的含义是：
    #   - 对于每个样本，将从 prompt 的最后一个 token（即 <eos>）之后，到序列倒数第二个 token（-2）之间的所有 token 置为 1（即有效参与奖励/损失计算）。
    #   - 这里 -1 表示不包括最后一个 token（通常是 <eos>），因为最后一个 token 是模型生成的结束符，不参与奖励计算。
    mask[:, query_tensors.size(1)-1:-1] = 1

    # 初始化 score_rew，用于存放每个 token 的奖励分数，初始为 0，shape 为 (batch_size, seq_len)
    score_reward = np.zeros(mask.shape)  # (bs, seq_len)

    # 这里将 correctness（即每个样本的奖励分数，通常是 0/1，表示答案是否正确）赋值给 score_rew 的倒数第二列
    # 原因如下：
    #   - 在大多数生成任务中，模型生成的最后一个 token（即 <eos> 之前的 token）才是完整答案的结束点，
    #     我们只对这个 token 给奖励（如正确为 1，错误为 0），而不是对整个序列的每个 token 都给奖励。
    #   - 因此，将 correctness 赋值到 score_rew 的倒数第二列（-2），
    #     就是把奖励分数分配到每个样本的最后一个有效输出 token 上（即 <eos> 之前的那个 token）。
    #   - 这样做可以确保 PPO 算法只在答案生成结束时给予奖励，符合 RLHF 的常见做法。
    score_reward[:, -2] = np.array(correctness)

    # 找到一个 batch 中每个序列的 <eos> 结束位置（即生成结束的位置），
    # nonzero 返回的是所有等于 <eos> 的 (batch_idx, token_idx) 坐标
    nonzero = (model_input_ids == tokenizer.eos_token_id).nonzero() 
    # ==========================

    for batch_idx, token_idx in nonzero:
        mask[batch_idx, token_idx:] = 0
        score_reward[batch_idx, token_idx:] = 0
        score_reward[batch_idx, token_idx-1] = correctness[batch_idx]


    
    # Make the kl reward and the full reward
    kl_reward = None
    reward = score_reward
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)   # 注意这里是 log(old) - log(ref), log(old/ref)<0  
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_reward = np.zeros(mask.shape)  # (bs, seqlen)
        kl_reward[:, :-1] = -kl     # NOTE the minus sign
 
        kl_coef = args["kl_coef"]
        reward = score_reward + kl_coef * kl_reward


    # Process value, return, advantage, logprob
    value = None
    gamma = args['gamma']
    lam = args['lam']
    advantage = np.zeros_like(reward)

    for i in range(len(reward)):
        cur_reward, cur_value = reward[i], value[i]
        cur_delta = 
        cur_advantage = 
        cur_advantage[:prompt_len-1] = 0
        advantage[i][:-1] = cur_advantage
    

    # lambda_return = GAE + values
    ret = advantage + value # (bs, seqlen)



    ## Debug
    # accelerator.print("padded_prompt_len:", prompt_len)
    # accelerator.print("model_input_ids:", tokenizer.batch_decode(model_input_ids[:1].cpu().numpy().tolist()))
    # accelerator.print("model_attention_mask:", model_attention_mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("mask:", mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("rew:", rew[:1].cpu().float().numpy().tolist())
    # accelerator.print("ret:", ret[:1].cpu().float().numpy().tolist())
    # accelerator.print("val:", val[:1].cpu().float().numpy().tolist())
    # accelerator.print("adv:", adv[:1].cpu().float().numpy().tolist())
    # accelerator.print("old_logprob:", old_logprob[:1].cpu().float().numpy().tolist())


    model.train()  # 切换回训练模式


    return model_input_ids, model_attention_mask, mask, reward, score_reward, kl_reward, ret, correctness, value, old_logprob, ref_logprob, advantage






def train_one_epoch(args, model, ref_model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader,
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    vf_coef = args['vf_coef']
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        for idx, batch in t:
            result_dict = defaultdict(list)
            # Do rollout first
            model.eval()
            model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv = rollout(
                args, model, ref_model, tokenizer,
                query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
                query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
                answer_values=batch['ppo_forward_kwargs']['answer_values'],
                src_name=train_dataset[0]['item_id'].split('_')[0],
            )
            model.train()
            # preprocess
            raw_adv = adv
            if args['adv_whitening'] == 'global':
                adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
            elif args['adv_whitening'] == 'local':
                adv = masked_whiten(adv, mask)

            batch_size_per_gpu = len(batch['ppo_forward_kwargs']['query'])
            mini_batch_size_per_gpu = args["mini_batch_size"]
            ppo_epochs = args["ppo_epochs"]
            train_stats = {}
            for _ in range(ppo_epochs):
                perms = torch.randperm(batch_size_per_gpu)
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                    # Subset to batch
                    cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                    cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
                    cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                    cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_score_rew = score_rew[b_inds].contiguous() # mini_bs x seqlen
                    cur_kl_rew = None if kl_rew is None else kl_rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                    cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_raw_adv = raw_adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_input_ids = model_input_ids[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_attention_mask = model_attention_mask[b_inds].contiguous()  # mini_bs x seqlen
                    
                    resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                    cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                    query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)

                    # Preprocess advantage and get metrics  
                    cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                    mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)

                    # Forward current model
                    model.eval()
                    lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                    logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)

                    # Compute losses
                    loss = 0

                    # policy gradient loss
                    ratio = torch.exp(logprob - cur_old_logprob)
                    pg_losses = -cur_adv[:, :-1] * ratio
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()
                    # pg_loss = (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum() / cur_mask[:, :-1].sum()
                    # pg_loss = (-logprob * cur_ret[:,:-1]).sum() / cur_mask[:, :-1].sum()

                    # value loss
                    vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                    # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                    # total loss
                    loss += pg_loss + vf_coef * vf_loss

                    # model_output = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
                    # logits = model_output[0]
                    # logprob_dist = torch.nn.functional.log_softmax(logits,dim=-1)
                    # logprob = torch.gather(logprob_dist, 2, model_input_ids[:, 1:].unsqueeze(2)).squeeze(-1)
                    # loss_pg = (-logprob * ret[:,:-1]).sum() / torch.maximum(torch.sum(mask[:,:-1]), torch.tensor(1.0))
                    # loss += loss_pg

                    # sft_model_input_ids = batch['ppo_forward_kwargs']['sft_model_input_ids']
                    # sft_model_attention_mask = batch['ppo_forward_kwargs']['sft_model_attention_mask']
                    # sft_model_labels = batch['ppo_forward_kwargs']['sft_model_labels']
                    # loss_sft = model(input_ids=sft_model_input_ids, attention_mask=sft_model_attention_mask, labels=sft_model_labels)[0]
                    # loss += loss_sft

                    # token related metrics
                    mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                    std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                    mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                    std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                    # value related metrics
                    # vf_expl_var_num = torch.var(torch.masked_select(cur_ret - vpreds, cur_mask.bool())) 
                    # vf_expl_var_dem = torch.var(torch.masked_select(cur_ret, cur_mask.bool()))
                    vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                    vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                    vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                    vf_expl_var = max(-1.0, vf_expl_var.item())  # the truncated value suffices
                    mean_vpred = masked_mean(vpreds, cur_mask)
                    mean_return = masked_mean(cur_ret, cur_mask)
                    mean_reward = masked_mean(cur_rew, cur_mask)
                    mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                    mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                    mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward

                    # policy related metrics
                    mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                    #mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                    mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
                    # sequence-level kl
                    mean_seq_kl = -1.0
                    if cur_kl_rew is not None:
                        cur_kl = -cur_kl_rew
                        seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                        mean_seq_kl = torch.mean(seq_kl)

                    # Update
                    epoch_result_dict['loss'].append(loss.item())

                    # accelerator.backward(loss)
                    # accelerator.deepspeed_engine_wrapped.backward(loss)
                    # runs backpropagation and handles mixed precision
                    if accelerator.distributed_type == "DEEPSPEED":
                        accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                        total_grad_norm = 0.0
                        for n, p in model.named_parameters():
                            cur_grad = deepspeed.utils.safe_get_full_grad(p).view(-1)
                            cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                            if cur_grad_norm_sqrt < 1e-8:
                                accelerator.print(f'{n} grad_norm_sqrt: {cur_grad_norm_sqrt}')
                            total_grad_norm += cur_grad_norm_sqrt ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        # Deepspeed's `engine.step` performs the following operations:
                        # - gradient accumulation check
                        # - gradient clipping
                        # - optimizer step
                        # - zero grad
                        # - checking overflow
                        # - lr_scheduler step (only if engine.lr_scheduler is not None)
                        accelerator.deepspeed_engine_wrapped.engine.step()
                    else:
                        accelerator.backward(loss)
                        #accelerator.backward(loss)
                        total_grad_norm = -1.0
                        if clip_grad_norm is not None:
                            total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                    # Update running stats
                    n_correct, total = do_gather([sum(correctness), len(correctness)])
                    train_stats["acc"] = n_correct / total
                    train_stats["ncor"] = n_correct
                    train_stats["total"] = total
                    train_stats['pg_loss'] = pg_loss.item()
                    train_stats['vf_loss'] = vf_loss.item()
                    train_stats['vf_expl_var'] = vf_expl_var

                    for k, v in train_stats.items():
                        result_dict[k].append(v)

                    total_param_norm = 0.0
                    if accelerator.distributed_type == "DEEPSPEED":
                        for n, p in model.named_parameters():
                            cur_param = deepspeed.utils.safe_get_full_fp32_param(p).view(-1)
                            total_param_norm += torch.norm(cur_param, 2) ** 2
                        total_param_norm = total_param_norm ** 0.5
                    else:
                        total_param_norm = torch.norm(
                            torch.cat([p.view(-1) for p in model.parameters()]),
                            p=2  # L2 norm
                        )
                    # logging
                    if accelerator.is_main_process and args['wandb_log']:
                        wandb.log({
                            "nn/total_grad_norm": total_grad_norm,
                            "nn/total_param_norm": total_param_norm,
                            "nn/lr": scheduler.get_last_lr()[0],
                        }, step=global_iter_num)
                        wandb.log({
                            "acc/acc": train_stats["acc"],
                            "acc/ncor": train_stats["ncor"],
                            "acc/total": train_stats["total"],
                        }, step=global_iter_num)
                        wandb.log({
                            "loss/loss:": loss,
                            "loss/pg_loss": pg_loss,
                            "loss/vf_loss": vf_loss,
                        }, step=global_iter_num)
                        wandb.log({
                            "tokens/mean_query_len": mean_query_len,
                            "tokens/std_query_len": std_query_len,
                            "tokens/mean_resp_len": mean_resp_len,
                            "tokens/std_resp_len": std_resp_len,
                        }, step=global_iter_num)
                        wandb.log({
                            "policy/mean_ratio": mean_ratio,
                            "policy/mean_adv": mean_adv,
                            "policy/var_adv": var_adv,
                            "policy/mean_logprob": mean_logprob,
                            "policy/mean_seq_kl": mean_seq_kl,
                        }, step=global_iter_num)
                        wandb.log({
                            "value/vf_expl_var": vf_expl_var,
                            "value/mean_vpred": mean_vpred,
                            "value/mean_return": mean_return,
                            "value/mean_reward": mean_reward,
                            "value/mean_score_reward": mean_score_reward,
                            "value/mean_kl_reward": mean_kl_reward,
                            "value/mean_kcxkl_reward": mean_kcxkl_reward,
                        }, step=global_iter_num)
                    # Update iter num
                    # torch.distributed.barrier()
                    global_iter_num += 1

            scheduler.step()
            global_step += 1
            # accelerator.empty_cache()
            # Step update metric
            epoch_result_dict['loss'].append(loss.item())
            for k, v in train_stats.items():
                epoch_result_dict[k].append(v)

            # Step evaluating
            eval_log_dict = {}
            is_best = False
            if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

            # Step logging
            train_log_dict = {}
            if logging_step_freq is not None and global_step % logging_step_freq == 0:
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            # Step saving
            if saving_step_freq is not None and global_step % saving_step_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f'best')
                    do_checkpoint(args, model, tokenizer, save_path)
                if args['keep_num_ckpt'] > 0:
                    save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

            # Keep only max_record items
            for k, v in epoch_result_dict.items():
                if len(v) > 1:
                    epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step, global_iter_num







def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []






def main(args):
    set_seed(args['seed'] + accelerator.process_index)






    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              test_dataloader)




    if ref_model is not None:
        if accelerator.distributed_type == 'DEEPSPEED':
            ref_model = prepare_deepspeed_ref_model(ref_model)
        else:
            ref_model = accelerator.prepare(ref_model)
    

    most_recent_ckpts_paths = []
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=False) as t:

        for epoch in t:
            kwargs = {

            }

            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)




if __name__ == '__main__':
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


    parser = HfArgumentParser(Arguments)


    (args,) = parser.parse_args_into_dataclasses()


    args = asdict(args)