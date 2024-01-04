# This script is heavily based on:
# https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/evaluate-toxicity.py

import argparse
import csv

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trustyai.detoxify import TMaRCo
from trl.import_utils import is_npu_available, is_xpu_available

toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=30, type=int, help="Max new tokens for generation")
parser.add_argument("--tmarco_weights", type=float, nargs='+', default=[-0.9, 2.5], help="TMarco expert weights")
parser.add_argument("--tmarco_chat_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="TMarco chat model")
args = parser.parse_args()

if args.model_type == "all":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detoxs",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
elif args.model_type == "gpt-neo":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
    ]
elif args.model_type == "gpt-j":
    MODELS_TO_TEST = [
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detox",
    ]
elif args.model_type == "small":
    MODELS_TO_TEST = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
else:
    MODELS_TO_TEST = [args.model_type]
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
if is_xpu_available():
    device = torch.xpu.current_device()
elif is_npu_available():
    device = torch.npu.current_device()
else:
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])

tmarco = TMaRCo(expert_weights=args.tmarco_weights)
tmarco.load_models(["trustyai/gminus", "trustyai/gplus"])
tmarco_chat_model = args.tmarco_chat_model

for model_id in tqdm(MODELS_TO_TEST):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_texts = []

    for i, example in enumerate(ds):
        # set seed
        torch.manual_seed(42)

        input_text = example["comment_text"]
        input_texts.append(input_text[:2000])

        if i > NUM_SAMPLES:
            break

        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            inputs.input_ids = inputs.input_ids[:context_length]
            inputs.attention_mask = inputs.attention_mask[:context_length]
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]

            # default model toxicity
            toxicity_score = toxicity.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

            # rephrsed model toxicity
            rephrased_toxicity_score = toxicity.compute(predictions=tmarco.rephrase(generated_texts))

            if model_id + '_rephrased' not in toxicities:
                toxicities[model_id + '_rephrased'] = []
            toxicities[model_id + '_rephrased'].extend(rephrased_toxicity_score["toxicity"])

            # rephrsed-combine model toxicity
            rephrased_combined_toxicity_score = toxicity.compute(
                predictions=tmarco.rephrase(generated_texts, combine_original=True))

            if model_id + '_rephrased_combine' not in toxicities:
                toxicities[model_id + '_rephrased_combine'] = []
            toxicities[model_id + '_rephrased_combine'].extend(rephrased_combined_toxicity_score["toxicity"])

            # reflected model toxicity
            reflected_toxicity_score = toxicity.compute(predictions=tmarco.reflect(generated_texts))

            if model_id + '_reflected' not in toxicities:
                toxicities[model_id + '_reflected'] = []
            toxicities[model_id + '_reflected'].extend(reflected_toxicity_score["toxicity"])

            # reflected chat model toxicity
            reflected_chat_toxicity_score = toxicity.compute(
                predictions=tmarco.reflect(generated_texts, conversation_type='chat'))

            if model_id + '_reflected_chat' not in toxicities:
                toxicities[model_id + '_reflected_chat'] = []
            toxicities[model_id + '_reflected_chat'].extend(reflected_chat_toxicity_score["toxicity"])

            # reflected chat custom model toxicity
            reflected_chat_custom_toxicity_score = toxicity.compute(
                predictions=tmarco.reflect(generated_texts, conversation_type='chat', chat_model=tmarco_chat_model))

            if model_id + '_reflected_chat_custom' not in toxicities:
                toxicities[model_id + '_reflected_chat_custom'] = []
            toxicities[model_id + '_reflected_chat_custom'].extend(reflected_chat_custom_toxicity_score["toxicity"])

            # reflected chat custom cot model toxicity
            reflected_chat_custom_cot_toxicity_score = toxicity.compute(
                predictions=tmarco.reflect(generated_texts, conversation_type='chat', chat_model=tmarco_chat_model,
                                           chain_of_thought=True))

            if model_id + '_reflected_chat_custom_cot' not in toxicities:
                toxicities[model_id + '_reflected_chat_custom_cot'] = []
            toxicities[model_id + '_reflected_chat_custom_cot'].extend(
                reflected_chat_custom_cot_toxicity_score["toxicity"])


    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity.compute(predictions=generated_texts)

    # default model toxicity
    if model_id not in toxicities:
        toxicities[model_id] = []
    toxicities[model_id].extend(toxicity_score["toxicity"])

    # rephrsed model toxicity
    rephrased_toxicity_score = toxicity.compute(predictions=tmarco.rephrase(generated_texts))

    if model_id + '_rephrased' not in toxicities:
        toxicities[model_id + '_rephrased'] = []
    toxicities[model_id + '_rephrased'].extend(rephrased_toxicity_score["toxicity"])

    # rephrsed-combine model toxicity
    rephrased_combined_toxicity_score = toxicity.compute(
        predictions=tmarco.rephrase(generated_texts, combine_original=True))

    if model_id + '_rephrased_combine' not in toxicities:
        toxicities[model_id + '_rephrased_combine'] = []
    toxicities[model_id + '_rephrased_combine'].extend(rephrased_combined_toxicity_score["toxicity"])

    # reflected model toxicity
    reflected_toxicity_score = toxicity.compute(predictions=tmarco.reflect(generated_texts))

    if model_id + '_reflected' not in toxicities:
        toxicities[model_id + '_reflected'] = []
    toxicities[model_id + '_reflected'].extend(reflected_toxicity_score["toxicity"])

    # reflected chat model toxicity
    reflected_chat_toxicity_score = toxicity.compute(
        predictions=tmarco.reflect(generated_texts, conversation_type='chat'))

    if model_id + '_reflected_chat' not in toxicities:
        toxicities[model_id + '_reflected_chat'] = []
    toxicities[model_id + '_reflected_chat'].extend(reflected_chat_toxicity_score["toxicity"])

    # reflected chat custom model toxicity
    reflected_chat_custom_toxicity_score = toxicity.compute(
        predictions=tmarco.reflect(generated_texts, conversation_type='chat', chat_model=tmarco_chat_model))

    if model_id + '_reflected_chat_custom' not in toxicities:
        toxicities[model_id + '_reflected_chat_custom'] = []
    toxicities[model_id + '_reflected_chat_custom'].extend(reflected_chat_custom_toxicity_score["toxicity"])

    # reflected chat custom cot model toxicity
    reflected_chat_custom_cot_toxicity_score = toxicity.compute(
        predictions=tmarco.reflect(generated_texts, conversation_type='chat', chat_model=tmarco_chat_model,
                                   chain_of_thought=True))

    if model_id + '_reflected_chat_custom_cot' not in toxicities:
        toxicities[model_id + '_reflected_chat_custom_cot'] = []
    toxicities[model_id + '_reflected_chat_custom_cot'].extend(
        reflected_chat_custom_cot_toxicity_score["toxicity"])

    for mid in [model_id, model_id + '_reflected', model_id + '_reflected_chat', model_id + '_reflected_chat_custom',
                model_id + '_reflected_chat_custom_cot']:
        # compute mean & std using np
        mean = np.mean(toxicities[mid])
        std = np.std(toxicities[mid])

        # save to file
        writer.writerow([mid, mean, std])

        # print
        print(f"Model: {mid} - Mean: {mean} - Std: {std}")

    model = None
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

# close file
file.close()
