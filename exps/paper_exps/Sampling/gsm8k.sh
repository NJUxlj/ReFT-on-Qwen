# gsm8k
## SDP 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama' \
ckpt_name='global_step_308_epoch_2' \
input_path='data/gsm8k_python_sdp.json' \
engine='python' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh






## NL 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama' \
ckpt_name='global_step_312_epoch_2' \
input_path='data/gsm8k_nl.json' \
engine='nl' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh