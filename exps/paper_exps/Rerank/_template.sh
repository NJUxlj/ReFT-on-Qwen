#!/bin/bash
export TOKENIZERS_PARALLELISM=True

### Required variables
# 下面几行是 Bash 的参数默认值赋值语法
# ${var:-default} 的意思是：如果变量 var 没有被设置（unset）或者为空（null），就使用 default 作为默认值
# 例如：exp_name=${exp_name:-''} 表示如果 exp_name 没有被设置，则赋值为空字符串
exp_name=${exp_name:-''}                # 如果 exp_name 没有设置，则默认为空字符串
train_file=${train_file:-''}            # 如果 train_file 没有设置，则默认为空字符串
test_file=${test_file:-''}              # 如果 test_file 没有设置，则默认为空字符串
model_name_or_path=${model_name_or_path:-''}      # 如果 model_name_or_path 没有设置，则默认为空字符串
tokenizer_name_or_path=${tokenizer_name_or_path:-''}  # 如果 tokenizer_name_or_path 没有设置，则默认为空字符串
n_epochs=${n_epochs:-''}                # 如果 n_epochs 没有设置，则默认为空字符串

### Default variables
model_dir="ppo_paper_final_new/_models_outputs_rerank/${exp_name}/"
config_file="./default_config_deepspeed.yaml"



batch_size="3"
gradient_accumulation_steps="1"
max_input_length="700"
num_workers="8"
learning_rate="1e-6"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"
keep_num_ckpt='0'


# ============= 日志打印和检查点保存  ===========

logging_epoch_freq="1"
evaluating_epoch_freq="1"
saving_epoch_freq="-100"

logging_step_freq="10"
evaluating_step_freq="1000"
saving_step_freq="-100"


###########

wandb_log="True"
wandb_project="ReFT"
wandb_run_name="${exp_name}"




########
num_processes='8'
main_process_port='8889'




mkdir -p "${model_dir}"

accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    train_reward_model.py \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --n_epochs "${n_epochs}" \
            --num_workers "${num_workers}" \
            --learning_rate "${learning_rate}" \
            --weight_decay "${weight_decay}" \
            --warmup_step "${warmup_step}" \
            --clip_grad_norm "${clip_grad_norm}" \
            --evaluating_epoch_freq "${evaluating_epoch_freq}" \
            --logging_epoch_freq "${logging_epoch_freq}" \
            --saving_epoch_freq "${saving_epoch_freq}" \
            --evaluating_step_freq "${evaluating_step_freq}" \
            --logging_step_freq "${logging_step_freq}" \
            --saving_step_freq "${saving_step_freq}" \
            --seed "${seed}" \
            --max_input_length "${max_input_length}" \
            --gradient_accumulation_steps "${gradient_accumulation_steps}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            --wandb_log "${wandb_log}" \
            --wandb_project "${wandb_project}" \
            --wandb_run_name "${wandb_run_name}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)


# 1> >(tee "${model_dir}"/"${exp_name}".log) \
# 2> >(tee "${model_dir}"/"${exp_name}".err >&2)

# 上面两行 bash 脚本的作用是将标准输出（stdout）和标准错误输出（stderr）分别重定向到不同的日志文件中，方便后续调试和查看训练过程中的信息。
        # 1> >(tee "${model_dir}"/"${exp_name}".log) \
        #   - 这行表示将标准输出（文件描述符1）通过 tee 命令同时输出到终端和 "${model_dir}/${exp_name}.log" 文件中。
        #   - 这样可以实时在终端看到输出内容，同时也会把内容写入日志文件，便于后续查阅。
        # 2> >(tee "${model_dir}"/"${exp_name}".err >&2)
        #   - 这行表示将标准错误输出（文件描述符2）通过 tee 命令同时输出到终端和 "${model_dir}/${exp_name}.err" 文件中。
        #   - tee 后面的 >&2 表示 tee 命令的输出也重定向到标准错误，这样终端上能实时看到错误信息，同时也会写入错误日志文件。