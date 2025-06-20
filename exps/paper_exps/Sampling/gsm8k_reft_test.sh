# gsm8k
## SDP 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_codellama_reft' \
ckpt_name='best' \
input_path='data/gsm8k_test_set.json' \
engine='python' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh







## NL 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_codellama_reft' \
ckpt_name='best' \
input_path='data/gsm8k_test_set.json' \
engine='nl' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh