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
from collections import Counter
from dataclasses import dataclass, field, asdict
from datasets import Dataset
from datetime import timedelta
import deepspeed
from functools import partial
import json
import os
from src.python_engine import run_python_code
from src.utils import set_seed, write_data, floatify
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict
import wandb
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


}

### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],



}

compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,


}


def tokenize_fn(examples:Dict, tokenizer, max_length, src_name, engine):
    features = {'input_ids':[], 'attention_mask': [], "answer_value": [], "answer_cot": [], "question": [], 'item_id': []}

    for idx, question in enumerate(examples['question']):
        text = f"{instruction}{question}{cot_trigger}"
        if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and engine == 'python':
            text += f'def solution():\n    """{question}"""\n'

        source_text_res = tokenizer.encode_plus(text, max_length=max_length, truncation=True, add_special_tokens=False)
        features["input_ids"].append(source_text_res["input_ids"])
        features['attention_mask'].append(source_text_res['attention_mask'])
        features['question'].append(question)
        features['answer_value'].append(examples['answer_value'][idx])
        features['answer_cot'].append(None if 'answer_cot' not in examples else examples['answer_cot'][idx])
        features['item_id'].append(examples['item_id'][idx])

    return features

def collate_fn():
    pass


def main(args):
    pass



if __name__ == '__main__':
    from transformers import HfArgumentParser

    NONE_INT = -100 
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name: str
        input_path: str
        save_dir: str
        engine: str
        batch_size: int=field(default=2)
        max_length: int=field(default=1024)
        num_return_sequences: int=field(default=1)
        temperature: float=field(default=1.0)
        do_sample: bool=field(default=False)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')

    parser = HfArgumentParser(Arguments)
    (args, ) = parser.parse_args_and_config()

    args = asdict(args)

    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None


    accelerator = Accelerator(InitProcessGroupKwargs(timeout = timedelta(seconds = 18000)))  # 最多等待 5 小时的预处理

    print(f"args: \n{json.dumps(args, indent=4)}")
    main(args)