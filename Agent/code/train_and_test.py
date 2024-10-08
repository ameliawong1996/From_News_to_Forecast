import os
import sys
import time

def start_a_new_train(
    exp_name,
    train_file_path,
    resume_from_checkpoint='None',
    result_saving_dir='{path_to_your_checkpoint}',
    learning_rate='1e-4',
    save_steps=400,
    epoch=1,
    prompter_name='ts_test',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=6,
    lora_r=8,
    lora_alpha=16,
):
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_model = os.path.join(result_saving_dir, exp_name)
    if os.path.isdir(output_model):
        raise ValueError(f'experiment {output_model} already existed.')
    log_path = os.path.join(output_model, f'{exp_name}-{formatted_time}.log')
    
    command_template = f'''
    deepspeed --include localhost:0 --master_port 29000 llm-finetune.py \
    --model_name_or_path [path_to_your_LLM:daryl149/llama-2-7b-chat-hf] \
    --tokenizer_name [path_to_your_LLM:daryl149/llama-2-7b-chat-hf] \
    --output_dir {output_model} \
    --resume_from_checkpoint {resume_from_checkpoint} \
    --train_files {train_file_path} \
    --prompter_name {prompter_name} \
    --per_device_train_batch_size {per_device_train_batch_size} \
    --do_train \
    --use_fast_tokenizer true \
    --learning_rate {learning_rate} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --num_train_epochs {epoch} \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r {lora_r} \
    --lora_alpha {lora_alpha} \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 12 \
    --save_steps {save_steps} \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 3072 \
    --report_to tensorboard \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --ddp_timeout 18000000 \
    '''
    os.system(command_template)
    
    
def run_evaluation(
    checkpoint,
    val_data,
    save_path,
    prompter_name='ts_test',
):
    command_template = f'''
    python validation.py \
    --base_model [path_to_your_LLM:daryl149/llama-2-7b-chat-hf] \
    --lora_weights {checkpoint} \
    --val_data {val_data} \
    --prompter_name ts_test \
    --save {save_path}
    '''
    os.system(command_template)