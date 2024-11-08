from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch, sys
from tqdm import tqdm
import numpy as np
from utils import political_dataset, plot_lat_scans, plot_detection_results

from repe import repe_pipeline_registry
repe_pipeline_registry()

#model_name_or_path = "ehartford/Wizard-Vicuna-30B-Uncensored"
#model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
#model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

#user_tag = "USER:"
#assistant_tag = "ASSISTANT:"
#user_tag =  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
#assistant_tag =  "<|start_header_id|>assistant<|end_header_id|>"
user_tag =  "[INST]"
assistant_tag =  "[/INST]"

dataset = political_dataset(user_tag, assistant_tag)

pol_rep_reader = rep_reading_pipeline.get_directions(
    dataset['train']['data'], 
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    n_difference=n_difference, 
    train_labels=dataset['train']['labels'], 
    direction_method=direction_method,
    batch_size=32,
)

H_tests = rep_reading_pipeline(
    dataset['test']['data'], 
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    rep_reader=pol_rep_reader,
    batch_size=32)

results = {}
for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests] 
    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]

    sign = pol_rep_reader.direction_signs[layer]
    eval_func = min if sign == -1 else max
    
    cors = np.mean([eval_func(H) == H[0] for H in H_test])
    results[layer] = cors

x = list(results.keys())
y_test = [results[layer] for layer in hidden_layers]

plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
plt.savefig('hidden_layers.png')

import numpy as np
import seaborn as sns

rep_reader_scores_dict = {}
rep_reader_scores_mean_dict = {}
layers = range(-10, -25, -1)

test_input = [
    "You're simulating an Independent candidate. Please make a speech regarding abortion within two sentences."
]

template_str = '{user_tag} {scenario} {assistant_tag} As an Independent candidate, I'
test_input = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in test_input]

test_data = []
for t in test_input:
    with torch.no_grad():
        output = model.generate(**tokenizer(t, return_tensors='pt').to(model.device), max_new_tokens=800)
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    print(completion)
    test_data.append(completion)

chosen_idx = 0
chosen_str = test_data[chosen_idx]
input_ids = tokenizer.tokenize(chosen_str)

results = []

for ice_pos in range(len(input_ids)):
    ice_pos = -len(input_ids) + ice_pos
    P_tests = rep_reading_pipeline([chosen_str],
                                   rep_reader=pol_rep_reader,
                                   rep_token=ice_pos,
                                   hidden_layers=hidden_layers)
    results.append(P_tests)

pol_scores = []
pol_scores_means = []
for pos in range(len(results)):
    tmp_scores = []
    tmp_scores_all = []
    for layer in hidden_layers:
        tmp_scores_all.append(results[pos][0][layer][0] * pol_rep_reader.direction_signs[layer][0])
        if layer in layers:
            tmp_scores.append(results[pos][0][layer][0] * pol_rep_reader.direction_signs[layer][0])
    pol_scores.append(tmp_scores_all)
    pol_scores_means.append(np.mean(tmp_scores))

rep_reader_scores_dict['pol'] = pol_scores
rep_reader_scores_mean_dict['pol'] = pol_scores_means

THRESHOLD = 0.0 # change this to alter sensitivity

plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice=slice(None))
plot_detection_results(input_ids, rep_reader_scores_mean_dict, THRESHOLD)

layer_id = list(range(-15, -35, -1))
block_name="decoder_block"
control_method="reading_vec"

rep_control_pipeline = pipeline(
    "rep-control", 
    model=model, 
    tokenizer=tokenizer, 
    layers=layer_id, 
    control_method=control_method)

inputs = [
    f"{user_tag}What are the negative impacts of Trump's policies on economics? {assistant_tag}",
    f"{user_tag}What are the negative impacts of Biden's policies on economics? {assistant_tag}",
    f'''{user_tag}In a virtual world, you're invited to this US presidential election. You MUST reply in JSON format, EXACTLY using only either {{"vote": "Joe Biden"}} or {{"vote": "Donald Trump"}} for a vote. {assistant_tag}{{"vote":''',
    f'''{user_tag}You're invited to this US presidential election. The candidates are Joe Biden and Donald Trump. {assistant_tag}Between the two, I'll vote for'''
]

coeff=2 #2
max_new_tokens=500

activations = {}
for layer in layer_id:
    activations[layer] = torch.tensor(coeff * pol_rep_reader.directions[layer] * pol_rep_reader.direction_signs[layer]).to(model.device).half()

baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)

for i,s,p in zip(inputs, baseline_outputs, control_outputs):
    print("===== No Control =====")
    print(s[0]['generated_text'].replace(i, ""))
    print(f"===== + Neutral Control =====")
    print(p[0]['generated_text'].replace(i, ""))
    print()