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
from typing import Any, Dict
import sys
sys.path.append("./")
from src.utils import timeout
import time
from tqdm import tqdm
import numpy as np
from pebble import ProcessPool
import sympy
import math
import copy

from typing import List, Dict

# 下面这行代码的作用是：从全局变量中提取 'sympy' 和 'math' 这两个库对象，构建一个名为 global_restricted 的字典，
# 其键为库名字符串，值为对应的库对象。这样可以在受限的执行环境中只允许访问这两个库，提升代码执行的安全性。
global_restricted = {lib: globals()[lib] for lib in ['sympy', 'math']}
# del global_restricted['sympy'].init_session
local_restricted = {}

def exec_code(code_piece, _global_vars, _local_vars):
    exec(code_piece, _global_vars, _local_vars)



def eval_code(code_piece, _global_vars, _local_vars):
    exec(code_piece, _global_vars, _local_vars)




def run(code_piece, expr):
    # 初始化全局变量和局部变量的字典，用于后续代码执行的命名空间隔离
    _global_vars, _local_vars = {}, {}

    # 将允许的库（这里只允许'sympy'和'math'）加入到全局变量字典中
    for lib in ['sympy', 'math']:
        _global_vars[lib] = global_restricted[lib]  # 只允许访问受限的库对象

        # 如果 local_restricted 中有该库，也加入到局部变量字典
        if lib in local_restricted:
            _local_vars[lib]= local_restricted[lib]
        
    # 这里 exec 的作用是先执行 code_piece（通常是函数定义，比如 def solution()），
    # 这样在 _global_vars/_local_vars 命名空间中就有了 solution 这个函数对象。
    # 随后 eval(expr)（比如 expr="solution()"）才能调用到刚刚定义的函数并返回结果。
    # 如果没有 exec，eval 时会找不到 solution。
    exec(code_piece, _global_vars, _local_vars)
    result = eval(expr, _global_vars, _local_vars)
    return result



def process_code(code_gen, truncate_first_return=False)->str:
    '''
    处理函数定义的字符串 e.g., def solution(): ...，去除无关内容，只保留函数定义和主体部分
    
    Args:
        code_gen (str): 代码生成器生成的代码
        truncate_first_return (bool): 是否截断第一个 return 语句
        
    Returns:
        处理后的函数定义字符串
    
    '''
    ## 处理黑名单关键词
    if 'sys.exit' in code_gen:
        code_gen = code_gen.replace('sys.exit', 'print')
    
    snippet = code_gen.split('\n')

    updated_code_snippet = ['import math', 'import sympy']

    # 对代码进行后处理
    for snippet_line in snippet:
        if snippet_line.startswith('def solution'):
            updated_code_snippet.append(snippet_line)
            continue

        # 这里判断如果当前行是空行（去除空白字符后为""），就 break 跳出循环
        # 这样做的原因通常是：有些生成的代码在函数定义后会有一个空行，空行后面的内容可能不是我们想要的代码主体（比如多余的注释、解释、或者模型生成的无关内容）
        # 因此遇到空行就 break，可以只保留函数定义和主体部分，去掉后面无关内容，保证代码的简洁和安全
        if snippet_line.strip() == "":
            break

        if truncate_first_return:
            if snippet_line == "    return result":
                break

        updated_code_snippet.append(snippet_line)

    updated_code_gen = '\n'.join(updated_code_snippet)

    return updated_code_gen





def run_python_code(programs:List[str], TIMEOUT:float, safe=True):
    is_single_program = False

    updated_programs = [process_code(code) for code in programs]
    if safe:
        # Safer -- executed code can't affect main code (e.g numpy.random.seed(...))
        # But it is slow ... 
        with ProcessPool(max_workers=8) as pool:
            futures = [pool.schedule(run, args=[code, 'solution()']) for code in updated_programs]
            results = []

            for i, f in tqdm(enumerate(futures), total=len(futures), disable=True):
                try:
                    res = f.result()
                except e:
                    print(str(e))
                    res = None
                
                results.append(res)

    else:
        results = []
        for code in tqdm(updated_programs, disable=True):
            with timeout(seconds= int(TIMEOUT)):
                try:
                    res = run(code_piece = code, expr='solution()' )
                except Exception as e:
                    print(str(e), code)
                    res = None

                results.append(res)

    if is_single_program:
        assert len(results) == 1, len(results)
        return results[0]
    
    return results




if __name__ == '__main__': 
    code = '''
    def solution():
        """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
        import time
        time.sleep(2)
        from sympy import init_session
        init_session()
        # raise
        clips_april = 48
        clips_may = clips_april / 2
        clips_total = clips_april + clips_may
        result = clips_total
        # import numpy as np
        # np.random.seed(42)
        # return np.random.randint(10)
        # np.random.seed(42)
        return result
    '''.strip()

    print(code)

    s= time.time()

    for i in tqdm(range(1)):
        res = run_python_code([code]*10, 2.5, True)

        print(res)

    print(time.time()-s)

    # 这行代码的作用是：使用numpy库的random模块生成一个[0, 10)之间的随机整数，并打印出来
    print(np.random.randint(10))
    print(f"Average answer = {sum() / len(res)}")