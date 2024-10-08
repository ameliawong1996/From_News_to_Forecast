output_model=results/your_experiment_name
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
deepspeed --include localhost:0 --master_port 29000 llm-finetune.py \
    --model_name_or_path /root/autodl-tmp/llama-models/daryl149/llama-2-7b-chat-hf \
    --tokenizer_name /root/autodl-tmp/llama-models/daryl149/llama-2-7b-chat-hf \
    --train_files /root/From_News_to_Forecast/ts_data/AULF_train_data_2019-2020_iteration_5.json \
    --validation_files  /root/From_News_to_Forecast/ts_data/AULF_test_data_2019-2020_iteration_5.json\
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 400 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 10 \
    --eval_steps 10 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --ddp_timeout 18000000
