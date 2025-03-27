
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# base_model = "meta-llama/Llama-3.2-3B-Instruct"
# base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# base_model = "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"
# base_model = "enferAI/DeepSeek-R1-Distill-Qwen-14B-FP8"

base_model = "Qwen/QwQ-32B-AWQ"
# base_model = "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"
# model = AutoModelForCausalLM.from_pretrained(base_model)
# print(model.get_memory_footprint())

# dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

import json
from datasets import Dataset

with open("tarot_qa_finetuning_dataset_cleaned.json", "r") as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(len(dataset['question']))

def formatting_prompts_func(example):
    output_texts = []

    system_prompt = "You are a tarot reading assistant, your goal is to generate the best response to user's choosen card in 4 sentences"
    for i in range(len(example['question'])):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example['question'][i]},
            {"role": "assistant", "content": example['answer'][i]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# response_template = " ### Answer:"
response_template = "<|im_start|>assistant\n"  # specifically for 
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
print(model.get_memory_footprint())

peft_config = LoraConfig(r=32,
                        lora_alpha=64,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                      )

import sys
import argparse

def main():
    # parser = argparse.ArgumentParser(description="Training arguments", allow_abbrev=False)


    # parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    # parser.add_argument("--batch_size", type=int, default=help="Batch size")

    # # Parse arguments
    # args = parser.parse_args()


    args = TrainingArguments(
        output_dir='qwen_sft_4b_v2',
        warmup_steps=1,
        num_train_epochs=5, # adjust based on the data size
        per_device_train_batch_size=4, # use 4 if you have more GPU RAM
        gradient_accumulation_steps=4,
        save_strategy="epoch", #steps
        logging_steps=100,
        optim="paged_adamw_32bit",
        learning_rate=2.5e-5,
        fp16=True,
        seed=42,
        # save_steps=50,  # Save checkpoints every 50 steps
        do_eval=False,   
        push_to_hub=True,              # Enable pushing to the Hub
        hub_model_id="ranxxx/sft_awq"
        )

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config
    )


    trainer.train()

    new_model = "sft_awq"

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)


if __name__ == "__main__":
    main()