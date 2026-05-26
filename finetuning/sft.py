import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = ""

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            context = data["text"]
            catagory = data["category"]
            label = data["output"]
            message = {
                "instruction": "你是一名资深教师，请你根据用户的要求完成相应任务",
                "input": f"文本:{context},类型选型:{catagory}",
                "output": label,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
            
            
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n你是一名资深教师，请你根据用户的要求完成相应任务<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response
    

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads() 

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"


train_data = pd.read_json("data/finetune_data.jsonl")
train_data = train_data.sample(frac=1).reset_index(drop=True)

train_ds = Dataset.from_pandas(train_data)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    inference_mode=False, 
    r=32,  
    lora_alpha=32,
    lora_dropout=0.1, 
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/GLM4-9B/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=1000,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
    
