import pickle
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from repe import repe_pipeline_registry
repe_pipeline_registry()

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1


user_tag =  "<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>"
assistant_tag =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

layer_id = list(range(-15, -20, -1))
block_name="decoder_block"
control_method="reading_vec"

rep_control_pipeline = pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer, 
    layers=layer_id, 
    control_method=control_method)

inputs = [
    f"{user_tag}Please respond to the following statement: Abortion, when the woman’s life is not threatened, should always be illegal.\n Options: Strongly disagree, Disagree, Neither, Agree, Strongly agree{assistant_tag}",
    f"{user_tag}Please respond to the following statement: In criminal justice, punishment should be more important than rehabilitation.\n Options: Strongly disagree, Disagree, Neither, Agree, Strongly agree{assistant_tag}",
    f"{user_tag}Please respond to the following statement: Taxes should be increased on the rich to provide for the poor.\n Options: Strongly disagree, Disagree, Neither, Agree, Strongly agree{assistant_tag}",
    f'''{user_tag}Hello! How are you doing? {assistant_tag}''',
    f'''{user_tag}What is the answer of 5C2? Options: (A) 10 (B) 20 (C) Neither {assistant_tag}''',
    f'''{user_tag}I'd like to give my girlfriend a bouquet of flowers. Which one between roses and lavenders do you think is better? {assistant_tag}''',
    f'''{user_tag}You're invited to this US presidential election. The candidates are Joe Biden and Donald Trump. {assistant_tag}Between the two, I'll vote for'''
]

max_new_tokens=1000
# Load activations from the file
with open('activations.pkl', 'rb') as f:
    activations = pickle.load(f)

baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

for i,s,p in zip(inputs, baseline_outputs, control_outputs):
    print("===== No Control =====")
    print(s[0]['generated_text'].replace(i, ""))
    print(f"===== + Neutral Control =====")
    print(p[0]['generated_text'].replace(i, ""))
    print()