# pip install math_verify # reward function
# pip install "trl>=0.15"
CONDA_HOME="YOUR CONDA_HOME"
CONDA_ENV="swift"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NPROC_PER_NODE=4

swift rlhf \
    --rlhf_type grpo \
    --model PATH_TO_MODEL \
    --reward_funcs external_llm_judge external_uni \
    --external_plugins rl_plugin.py \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset PATH_TO_DATASET \
    --gradient_checkpointing true \
    --lazy_tokenize true \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $(expr 8 / $NPROC_PER_NODE) \
    --per_device_eval_batch_size $(expr 8 / $NPROC_PER_NODE) \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 1000 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /output \
    --warmup_ratio 0.02 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1 \
    --system data/system.txt \
    --log_completions true \
    --model_type glm4\
    --template glm4 \
