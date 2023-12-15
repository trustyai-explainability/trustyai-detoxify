# -*- coding: utf-8 -*-
import math
import itertools
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from scipy.spatial.distance import jensenshannon
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline, Conversation
from torch.nn.functional import softmax
from torch import Tensor, topk
import re


class TMaRCo:
    base = None
    experts = []
    expert_weights = []
    tokenizer = None

    def __init__(self, base=None, expert_weights=None, tokenizer=None, base_model="facebook/bart-large"):
        if expert_weights is None:
            expert_weights = [-0.5, 0.5]

        if tokenizer is None:
            self.tokenizer = BartTokenizer.from_pretrained(base_model, is_split_into_words=True,
                                                           add_prefix_space=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if base is not None:
            self.base = base
        else:
            self.base = BartForConditionalGeneration.from_pretrained(base_model, max_length=150,
                                                                     forced_bos_token_id=self.tokenizer.bos_token_id)

        self.expert_weights = expert_weights

    def load_models(self, expert_paths: list, expert_weights: list = None):
        if expert_weights is not None:
            self.expert_weights = expert_weights
        for expert_path in expert_paths:
            self.experts.append(
                BartForConditionalGeneration.from_pretrained(expert_path,
                                                             forced_bos_token_id=self.tokenizer.bos_token_id))

    def tokenize_function(self, examples, target_feature="comment_text"):
        return self.tokenizer(examples[target_feature], max_length=1024, truncation=True)

    @staticmethod
    def group_texts(examples, block_size=128):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def mask_tokens(self, sentence, mask_token):
        masked_sentences = []
        tokens = self.tokenizer.tokenize(sentence)
        for idx in range(len(tokens)):
            masked_sentence = tokens.copy()
            masked_sentence[idx] = mask_token
            masked_sentence = self.tokenizer.convert_tokens_to_string(masked_sentence)
            masked_sentences.append(masked_sentence)
        return masked_sentences

    def train_models(self, dataset_name: str = 'jigsaw_toxicity_pred', perc: int = 100, expert_feature: str = 'toxic',
                     data_dir: str = 'jigsaw-toxic-comment-classification-challenge', td_columns=None,
                     base_model="facebook/bart-large"):

        if td_columns is None:
            td_columns = ["comment_text", 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                          'identity_hate']

        ds_size = ['train[:' + str(perc) + '%]', 'test[:' + str(perc) + '%]']

        datasets_split = load_dataset(dataset_name, data_dir=data_dir, split=ds_size)
        datasets_split = DatasetDict({'train': datasets_split[0], 'test': datasets_split[1]})

        toxic_datasets = datasets_split.filter(lambda x: int(x[expert_feature]) == 1)

        tokenized_datasets = toxic_datasets.map(self.tokenize_function, batched=True, num_proc=4,
                                                remove_columns=td_columns)

        lm_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            batch_size=100,
            num_proc=4,
        )

        gminus = BartForConditionalGeneration.from_pretrained(base_model,
                                                              forced_bos_token_id=self.tokenizer.bos_token_id)

        training_args = TrainingArguments(
            "gminus",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=gminus,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"G- perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.save_model('gminus')

        # train gplus model
        nontoxic_datasets = datasets_split.filter(lambda x: int(x[expert_feature]) == 0)

        nontoxic_tokenized_datasets = nontoxic_datasets.map(self.tokenize_function, batched=True, num_proc=4,
                                                            remove_columns=td_columns)

        nontoxic_lm_datasets = nontoxic_tokenized_datasets.map(
            self.group_texts,
            batched=True,
            batch_size=100,
            num_proc=4,
        )

        gplus = BartForConditionalGeneration.from_pretrained(base_model,
                                                             forced_bos_token_id=self.tokenizer.bos_token_id)

        nt_training_args = TrainingArguments(
            "gplus",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=2,
        )

        nt_trainer = Trainer(
            model=gplus,
            args=nt_training_args,
            train_dataset=nontoxic_lm_datasets["train"],
            eval_dataset=nontoxic_lm_datasets["test"],
        )

        nt_trainer.train()

        nt_eval_results = nt_trainer.evaluate()
        print(f"G+ perplexity: {math.exp(nt_eval_results['eval_loss']):.2f}")
        print('training finished')

        nt_trainer.save_model('gplus')
        self.experts = [gminus, gplus]

    def mask(self, sentence: str, threshold: float = 1.2, normalize: bool = True, use_logits: bool = True,
             scores: list = None):
        if scores is None:
            scores = self.score(sentence, use_logits=use_logits, normalize=normalize)
        tokens = self.tokenizer.tokenize(sentence)
        masked_output = []
        for idx in range(len(tokens)):
            if scores[idx] > threshold:
                masked_output.append(self.tokenizer.mask_token)
            else:
                masked_output.append(tokens[idx])
        masked_sentence = self.tokenizer.convert_tokens_to_string(masked_output)
        return masked_sentence

    @staticmethod
    def compute_mask_probs(fmp, text_sentence):
        fm_result = fmp(text_sentence)
        score_list = [(d['token'], d['score']) for d in fm_result]
        token_scores = sorted(score_list, key=lambda x: x[0])
        token_ids = [e[1] for e in token_scores]
        token_scores = [e[1] for e in token_scores]
        return token_ids, token_scores

    def rephrase(self, original, masked_output, compute_probs: bool = False, verbose: bool = False,
                 combine_original: bool = False, expert_weights: list = None):
        if expert_weights is None:
            expert_weights = self.expert_weights
        base_logits = self.compute_mask_logits(self.base, original, mask=False)
        rephrased_tokens_ids = []
        masked_sentence_tokens = self.tokenizer.tokenize(masked_output)
        if combine_original:
            original_sentence_tokens = self.tokenizer(original)["input_ids"][1:-1]
        fmp_experts = []
        if compute_probs:
            for expert in self.experts:
                fmp_experts.append(pipeline("fill-mask", model=expert, tokenizer=self.tokenizer,
                                            top_k=self.tokenizer.vocab_size))
        for idx in range(len(masked_sentence_tokens)):
            if masked_sentence_tokens[idx] == self.tokenizer.mask_token:
                next_token_logits = base_logits[0, 1 + idx]
                if verbose:
                    self.print_token(next_token_logits)
                expert_logits = []
                current_sentence_ids = rephrased_tokens_ids + [self.tokenizer.mask_token_id]
                if combine_original and idx < len(masked_sentence_tokens) - 2:
                    current_sentence_ids = current_sentence_ids + original_sentence_tokens[idx + 1:]
                    if verbose:
                        print(self.tokenizer.convert_ids_to_tokens(current_sentence_ids))
                if compute_probs:
                    masked_sentence = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(current_sentence_ids))
                    for expert in fmp_experts:
                        _, scores = self.compute_mask_probs(expert, masked_sentence)
                        expert_logits.append(scores)
                    for eidx in range(len(expert_logits)):
                        tensor = Tensor(expert_logits[eidx])
                        next_token_logits *= expert_weights[eidx] * tensor
                    log_prob = next_token_logits
                else:
                    masked_sequence = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(current_sentence_ids))
                    eidx = 0
                    for expert in self.experts:
                        next_token_logits += expert_weights[eidx] * self.compute_mask_logits(expert, masked_sequence)
                        eidx += 1
                    log_prob = next_token_logits
                if verbose:
                    self.print_token(next_token_logits)
                argmaxed = np.argmax(log_prob).item()
                rephrased_token_id = argmaxed
                rephrased_tokens_ids.append(rephrased_token_id)
            else:
                rephrased_tokens_ids.append(self.tokenizer._convert_token_to_id(masked_sentence_tokens[idx]))
        return self.tokenizer.decode(rephrased_tokens_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    def print_token(self, token_logits):
        log_prob = softmax(token_logits, dim=0)
        argmaxed = np.argmax(log_prob).item()
        rephrased_token = self.tokenizer.decode(argmaxed)
        print(rephrased_token)
        print([self.tokenizer.decode(i.item()).strip() for i in
               topk(token_logits, 5)[1]])

    def compute_mask_logits(self, model, sequence, verbose: bool = False, mask: bool = True):
        if verbose:
            print(f'input sequence: {sequence}')
        subseq_ids = self.tokenizer(sequence, return_tensors="pt")
        if verbose:
            raw_outputs = model.generate(**subseq_ids)
            print(sequence)
            print(self.tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)[0])
        with torch.no_grad():
            if mask:
                mt_idx = torch.nonzero(subseq_ids.input_ids[0] == self.tokenizer.mask_token_id).item()
                return model.forward(**subseq_ids).logits[0, mt_idx]
            else:
                return model.forward(**subseq_ids).logits

    def compute_mask_logits_multiple(self, model, sequences, verbose: bool = False, mask: bool = True):
        if verbose:
            print(f'input sequences: {sequences}')
        subseq_ids = self.tokenizer(sequences, return_tensors="pt", padding=True)
        if verbose:
            raw_outputs = model.generate(**subseq_ids)
            print(sequences)
            print(self.tokenizer.batch_decode(raw_outputs, skip_special_tokens=True))
        with torch.no_grad():
            if mask:
                raw_outputs = model.forward(**subseq_ids).logits
                # TODO: speed up tensor aggregation below
                mt_idx = torch.nonzero(subseq_ids.input_ids == self.tokenizer.mask_token_id)[:, 1]
                tensors = []
                for idx in range(len(mt_idx)):
                    tensors.append(raw_outputs[idx, mt_idx[idx]].unsqueeze(0))
                return torch.cat(tensors, dim=0)
            else:
                return model.forward(**subseq_ids).logits

    def score(self, sentence, use_logits: bool = True, normalize: bool = True, verbose: bool = False):
        # TODO : verify presence of pad_token
        masked_sentences = self.mask_tokens(sentence, self.tokenizer.pad_token + self.tokenizer.mask_token)
        distributions = []
        for model in self.experts:
            if use_logits:
                logits = self.compute_mask_logits_multiple(model, masked_sentences, verbose=verbose)
                mask_substitution_scores = softmax(logits, dim=1)
            else:
                mask_substitution_scores = []
                fmp = pipeline("fill-mask", model=model, tokenizer=self.tokenizer, top_k=10)
                for masked_sentence in masked_sentences:
                    # approximated probabilities for top_k tokens
                    distr = fmp(masked_sentence)
                    mask_substitution_score = [x['score'] for x in distr]
                    mask_substitution_scores.append(mask_substitution_score.numpy())
            distributions.append(mask_substitution_scores)
        distr_pairs = itertools.combinations(distributions, 2)
        js_distances = []
        for distr_pair in distr_pairs:
            js_distance = jensenshannon(distr_pair[0], distr_pair[1], axis=1)
            if normalize:
                js_distance = js_distance / np.average(js_distance)
            js_distances.append(js_distance)
        js_distance = np.average(js_distances, axis=0)
        return js_distance

    def rephrase_incrementally(self, text, threshold=1.2, verbose=False, max_attempts=10):
        incrementally_rephrased = text
        idx = 1
        while True:
            if len(text.strip()) == 0:
                break
            scores = self.score(text, verbose=verbose)
            if all(scores) < threshold:
                break
            masked = self.mask(text, scores=scores, threshold=threshold)
            incrementally_rephrased = self.rephrase(text, masked, verbose=verbose)
            if verbose:
                print(f'step{idx}: {incrementally_rephrased}')
            if text == incrementally_rephrased or idx == max_attempts:
                break
            else:
                text = incrementally_rephrased
                idx += 1
        return incrementally_rephrased

    def to_dict(self, t, s):
        sd = dict()
        ts = self.tokenizer.tokenize(t)
        for idx in range(len(ts)):
            sd[ts[idx][1:]] = s[idx]
        return sd

    def reflect(self, input, threshold=1.2, verbose=False):
        scores = self.score(input)
        mask = self.mask(input, scores=scores, threshold=threshold)
        rephrased = self.rephrase(input, mask)
        scores_dict = self.to_dict(input, scores)
        bad_words = []
        for t, s in scores_dict.items():
            if s > threshold:
                bad_words.append("'" + t + "'")
        scored_words_text = ', '.join(bad_words)
        reflection_prompt = f"Detoxify '{input}'. "
        rephrase_command = "You may rephrase it as"
        reflection_prompt += f"The following words are toxic: {scored_words_text}. "
        reflection_prompt += rephrase_command + f" '{rephrased}'."
        converse_pipeline = pipeline("conversational", model=self.base, tokenizer=self.tokenizer)
        reflection_conversation = Conversation(reflection_prompt)
        reflected_output = converse_pipeline([reflection_conversation])
        if verbose:
            print(f'{reflected_output}')
        reflected_texts = []
        for generated_response in reflected_output.generated_responses:
            # TODO: proper parsing needed
            if rephrase_command in generated_response:
                gras = generated_response[generated_response.index(rephrase_command) + len(rephrase_command):]
                reflected_texts.append(gras)
            else:
                reflected_texts.append(generated_response)
        return reflected_texts
