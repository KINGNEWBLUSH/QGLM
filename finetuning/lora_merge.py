from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from peft import PeftModel
import torch
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import shutil

WEIGHT_PATH = 'weight/QGLM_GLM_RL_LORA'
TARGET_PATH = "weight/QGLM_GLM_RL"

# replace the config.json file with modified tokenizer and modeling files
tokenizer_dir = "data/model_meta"
 
 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    tokenizer.save_pretrained('./output/merge_result')
    model.save_pretrained(output_path)



if os.path.exists(TARGET_PATH):
    abs_target = os.path.abspath(TARGET_PATH)
    if abs_target in ("/", "") or len(abs_target) <= 1:
        raise RuntimeError(f"Refusing to delete unsafe path: {abs_target}")
    print(f"Clearing target path: {abs_target}")
    shutil.rmtree(abs_target)
os.makedirs(TARGET_PATH, exist_ok=True)

apply_lora("weight/glm-4-9b-chat", TARGET_PATH, WEIGHT_PATH)

# Replace the config.json file
target_path = "weight/QGLM_GLM_RL"


# os.makedirs(target_path, exist_ok=True)

trasfer_files = ["tokenizer_config.json", "tokenization_chatglm.py", "modeling_chatglm.py"]
for root, dirs, files in os.walk(tokenizer_dir):
    rel_root = os.path.relpath(root, tokenizer_dir)
    dest_root = target_path if rel_root == "." else os.path.join(target_path, rel_root)
    os.makedirs(dest_root, exist_ok=True)
    for fname in files:
        if fname not in trasfer_files:
            continue
        src_file = os.path.join(root, fname)
        dest_file = os.path.join(dest_root, fname)
        shutil.copy2(src_file, dest_file)
