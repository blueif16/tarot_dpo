from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from dotenv import load_dotenv
import os

load_dotenv()

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit")
# Load the adapter onto the base model
model = PeftModel.from_pretrained(base_model, "ranxxx/sft_qwen")

tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit")
train_dataset = load_dataset(os.getenv("PREF_REPO_ID"))

from trl import DPOTrainer, DPOConfig
training_args = DPOConfig(output_dir="qwen_dpo_v2", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()