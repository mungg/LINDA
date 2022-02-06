#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import random
import logging
import math
import nltk
import os
import copy
import datetime
import sys
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

from filelock import FileLock
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed, BartTokenizer,
)
from transformers.file_utils import is_offline_mode
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu

# sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from LINDA.data.dataset import create_examples, create_datasets
    from LINDA.modeling import BartForLINDA # todo: modeling_debug instead of modeling
    from LINDA.utils import extract_pairs, mask_tokens, mask_words
except:
    from data.dataset import create_examples, create_datasets
    from modeling import BartForLINDA # todo: modeling_debug instead of modeling
    from utils import extract_pairs, mask_tokens, mask_words

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')

log_file = 'output' + str(datetime.datetime.now()) + '.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.path.append('.')
    nltk.data.find("/path/you/need/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_generation", action='store_true')
    parser.add_argument("--do_finetuning", action='store_true')
    parser.add_argument("--gpu_ids", type=str, default="1")
    parser.add_argument("--train_mode", type=str, default='full',
                        help="'full' or 'tiny'. If set to 'tiny', trained with only 100 sentence pairs.")

    # other frequently used training settings
    parser.add_argument("--lam", type=float, default=0.001, help="lambda for L2 regularization in eq (7).")
    parser.add_argument("--fix_pair", type=bool, default=False, help="to fix sentence pair ordering (for overfitting)")
    parser.add_argument("--fix_alpha", type=bool, default=False, help="if fixed, set to 0.5 (for overfitting)")
    parser.add_argument("--add_gaussian_noise", type=bool, default=True, help="to add gaussian noise")
    parser.add_argument("--use_l2_loss", type=bool, default=True, help="to use L2 loss")
    parser.add_argument("--alpha_for_beta", type=float, default=0.0, help="0: uniform, above 0: beta distribution with alpha value")
    parser.add_argument("--input_noise_prob", type=float, default=0.0, help="probability for input noise(mask or dropout)")
    # resume training
    parser.add_argument(
        "--continue_training_from_ckpt", type=bool,
        default=False,
        help="whether to continue training the model from the ckpt or not"
             "if true, need to provide my_checkpoint"
    )
    parser.add_argument(
        "--my_checkpoint", type=str,
        help="set ckpt path if resuming training from the ckpt, activated only when continue_training_from_ckpt is true"
    )

    # dataset path and loading script
    parser.add_argument("--output_dir", type=str, default="./ckpt")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--gen_data_file_path", required=False, type=str)


    parser.add_argument(
        "--wiki_loading_script_path", type=str,
        default='./data/dataset_loading_script.py',
        help="data loading script for HF datasets"
    )

    # original arguements
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    # for eval
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    # model path and type
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/bart-large",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    # batch size and other hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # logging loss and saving model steps
    parser.add_argument(
        "--loss_logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--model_saving_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--train", default=None, help="dataset for training")

    args = parser.parse_args()
    return args



def train(args, tokenizer, model):
    writer = SummaryWriter(os.path.join("runs_new",args.output_dir.split("/")[-1]))

    if args.train_mode == "full":
        args.data_file_path = os.path.join(args.dataset_dir,"raw/train.txt")
    elif args.train_mode == 'tiny':
        args.data_file_path = os.path.join(args.dataset_dir, "raw/train_tiny.txt")

    logger.info("Creating train_dataset from {}".format(args.data_file_path))
    train_dataset = load_dataset(args.wiki_loading_script_path,
                                 data_files=args.data_file_path,
                                 split='train',
                                 cache_dir=args.train_dataset_cached_dir)
    logger.info("Tokenizing train_dataset")

    if args.input_noise_prob > 0.0:
        logger.info("Transforming sentence with masking")
        # Tokenizing for masked sentence
        def encode(examples):
            return tokenizer(mask_words(examples['sent_1'], args.input_noise_prob, tokenizer=tokenizer, is_span=True),
                             max_length=args.max_length, truncation=True, padding='max_length')

        train_dataset = train_dataset.map(encode,batched=True, batch_size=100000)
        train_dataset = train_dataset.rename_column('input_ids', 'masked_input_ids')
        train_dataset = train_dataset.rename_column('attention_mask', 'masked_attention_mask')

        logger.info("Transforming original sentence")

        # Tokenizing for non-masked sentence
        train_dataset = train_dataset.map(lambda examples: tokenizer(examples['sent_1'],
                                                                     max_length=args.max_length,
                                                                     padding='max_length',
                                                                     truncation=True),
                                          batched=True, batch_size=100000)
        total_train_batch_size = args.per_device_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask',
                                                        'masked_input_ids', 'masked_attention_mask', ])
    else:
        train_dataset = train_dataset.map(lambda examples: tokenizer(examples['sent_1'],
                                                                     max_length=args.max_length,
                                                                     padding='max_length',
                                                                     truncation=True),
                                          batched=True, batch_size=100000)
        total_train_batch_size = args.per_device_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
        train_dataset.set_format(type='torch',columns=['input_ids', 'attention_mask',])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=total_train_batch_size, shuffle=True)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model = torch.nn.DataParallel(model)
    model.to(args.device)

    # Check if saved optimizer or scheduler states exist
    if args.continue_training_from_ckpt and \
            os.path.isfile(os.path.join(args.my_checkpoint, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.my_checkpoint, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.my_checkpoint, "optimizer.pt")))
        lr_scheduler.load_state_dict(torch.load(os.path.join(args.my_checkpoint, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    total_loss = 0

    # Check if continuing training from a checkpoint
    if args.continue_training_from_ckpt and os.path.exists(args.my_checkpoint):
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = args.my_checkpoint.split("-")[-1].split("/")[0]  # step
        completed_steps = int(checkpoint_suffix)  # training for only 1 epoch, step == gloabl_step
        steps_trained_in_current_epoch = completed_steps % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from global step %d", completed_steps)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # subset train_dataset and reload train_dataloader
        train_dataset = Subset(train_dataset, range(total_train_batch_size * completed_steps, len(train_dataset)))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=total_train_batch_size, shuffle=False)

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # 1 -> -100 in labels
            label_1 = copy.deepcopy(batch['input_ids'])
            #label_1 = batch['input_ids'].new_zeros(batch['input_ids'].shape)
            #label_1[:, :-1] = batch['input_ids'][:, 1:].clone()
            #label_1[:, -1] = 1
            pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            label_1[label_1 == pad_token_idx] = -100

            #batch['input_ids'] = mask_tokens(batch['input_ids'].clone(), tokenizer, args.input_noise_prob)

            inputs = {
                "input_ids": batch['input_ids'],
                "attention_mask": batch['attention_mask'],
                "masked_input_ids": batch["masked_input_ids"] if args.input_noise_prob else None,
                "masked_attention_mask": batch["masked_attention_mask"] if args.input_noise_prob else None,
                "labels": label_1,
                "L2_reg": args.lam,
                "use_l2_loss": args.use_l2_loss,
                "fix_pair": args.fix_pair, # if set to true, always same pairs
                "fix_alpha": args.fix_alpha, # if set to true, alpha is fixed to 0.5 by default
                "add_gaussian_noise": args.add_gaussian_noise,
                "alpha_for_beta": args.alpha_for_beta,
            }
            outputs = model(**inputs)
            loss, lm_loss, l2_loss = outputs

            if args.n_gpu > 1:
                loss = loss.mean()
                lm_loss = lm_loss.mean()
                if l2_loss is not None:
                    l2_loss = l2_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                lm_loss = lm_loss / args.gradient_accumulation_steps
                if l2_loss is not None:
                    l2_loss = l2_loss / args.gradient_accumulation_steps

            loss.backward()
            writer.add_scalar("Loss/total", loss, completed_steps)
            writer.add_scalar("Loss/LM", lm_loss, completed_steps)
            if l2_loss is not None:
                writer.add_scalar("Loss/L2", l2_loss, completed_steps)

            total_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                writer.add_scalar("Learning rate/step", lr_scheduler.get_last_lr()[0], completed_steps)
                completed_steps += 1


            if step != 0 and step % args.loss_logging_steps == 0:
                writer.flush()
                if l2_loss is not None:
                    logger.info(" [STEP {}] avg loss: {:.3f}, batch loss: {:.3f}, lm loss: {:.2f}, l2 loss: {:.2f}".format(
                        completed_steps, (total_loss / completed_steps), loss.item(), lm_loss.item(), l2_loss.item()))
                else:
                    logger.info(" [STEP {}] avg loss: {:.3f}, batch loss: {:.3f}".format(
                        completed_steps, (total_loss / completed_steps), loss.item()))

            if step != 0 and args.model_saving_steps > 0 and step % args.model_saving_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(completed_steps))
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        # save model after each epoch
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(completed_steps))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        writer.close()


def evaluate(args, tokenizer, model):
    logger.info("***** Running evaluation *****")

    eval_output_path = os.path.join(args.dataset_dir, "eval_output_all.tsv")
    args.dev_data_file_path = os.path.join(args.dataset_dir, "raw/dev.txt")

    dev_sent_pairs = {}
    with open(args.dev_data_file_path, encoding='utf-8') as f:
        dev_sents = f.read().split("\n")
        dev_sents = dev_sents[:1000]

    logger.info("dev.txt contains 50,000 sentences.")
    logger.info("only {} sentences out of 50,000 sentences are being evaluated".format(len(dev_sents)))
    for i, sent in enumerate(dev_sents):
        dev_sent_pairs[i] = {}
        dev_sent_pairs[i]["sent_1"] = sent
        dev_sent_pairs[i]["sent_2"] = dev_sents[len(dev_sents) - i - 2]

    model.to(args.device)
    model.eval()

    model.config.output_past = True
    model.config.num_beams = 4

    total_pairs = 0

    fw = open(eval_output_path,"w")

    fw.write("{}\t{}\n".format(dev_sent_pairs[i]["sent_1"], dev_sent_pairs[i]["sent_2"]))

    for k, v in tqdm(dev_sent_pairs.items()):

        inputs_1 = tokenizer([v["sent_1"]], max_length=args.max_length, return_tensors='pt', padding='max_length', truncation=True)
        inputs_2 = tokenizer([v["sent_2"]], max_length=args.max_length, return_tensors='pt', padding='max_length', truncation=True)

        model.config.min_length = min(len(inputs_1), len(inputs_2))

        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)

        alpha_steps = np.arange(0, 1.1, 0.1)

        fw.write("{}\t{}\n".format(v["sent_1"], v["sent_2"]))
        for alpha in alpha_steps:

            encoder_outputs, target_len, (sentence_embed_1, sentence_embed_2) = model.get_noisy_embedding(
                input_ids_1=inputs_1["input_ids"],
                attention_mask_1=inputs_1["attention_mask"],
                input_ids_2=inputs_2["input_ids"],
                attention_mask_2=inputs_2["attention_mask"],
                alpha=alpha,
                sent_emb=True,
                mix_style=False)

            #similarity = cosine_similarity(sentence_embed_1, sentence_embed_2).detach().cpu().numpy()[0]
            similarity = (sentence_embed_1 -sentence_embed_2).pow(2).sum(dim=2).sqrt().mean().detach().cpu().item()
            model_kwargs = {"encoder_outputs": encoder_outputs}

            sample_output = model.generate(input_ids=None, **model_kwargs, max_length=128)
            generated_sent = tokenizer.decode(sample_output[0], skip_special_tokens=True)

            dev_sent_pairs[k]["generated"] = generated_sent

            reference_sent_1 = [tokenizer.convert_ids_to_tokens(inputs_1["input_ids"][0], skip_special_tokens=True)]
            reference_sent_2 = [tokenizer.convert_ids_to_tokens(inputs_2["input_ids"][0], skip_special_tokens=True)]

            candidate = tokenizer.convert_ids_to_tokens(sample_output[0], skip_special_tokens=True)

            bleu_sent_1 = sentence_bleu(reference_sent_1, candidate, weights=(1.0, 0.0, 0, 0))
            bleu_sent_2 = sentence_bleu(reference_sent_2, candidate, weights=(1.0, 0.0, 0, 0))

            fw.write("{}\t{}\t{}\t{}\t{}\n".format(alpha, generated_sent, bleu_sent_1, bleu_sent_2, similarity))

            total_pairs += 1

    fw.close()
    logger.info("**************************************")

def generation(args, tokenizer, model):
    logger.info("***** Running generation *****")

    gen_output_path = os.path.join(args.dataset_dir, "eval_{}_0.5_e_dist".format(args.train))
    gen_data_file_path = os.path.join(args.dataset_dir, args.train)

    with open(gen_data_file_path, encoding='utf-8') as f:
        datas = f.read().split("\n")
        gen_sents = [data.strip().split("\t")[-1] for data in datas]

    logger.info(f"{args.train} contains {len(gen_sents)} sentences.")

    indices = list(range(0, len(gen_sents)))
    random.shuffle(indices)

    model.to(args.device)
    model.eval()

    model.config.output_past = True
    model.config.num_beams = 4

    fw = open(gen_output_path, "w")

    for idx in tqdm(range(len(gen_sents))):

        sent_1_idx = idx
        sent_2_idx = idx  # To handle error when there is no other indices left excpet own value

        for index in indices:
            if idx != index:
                sent_2_idx = index
                indices.remove(index)
                break

        if sent_1_idx == sent_2_idx:
            logger.info("same index. stop generation")
            break

        sent_1 = gen_sents[sent_1_idx]
        sent_2 = gen_sents[sent_2_idx]

        fw.write("{}\t{}\n".format(sent_1, sent_2))

        inputs_1 = tokenizer([sent_1], max_length=args.max_length, return_tensors='pt', padding='max_length',
                             truncation=True)
        inputs_2 = tokenizer([sent_2], max_length=args.max_length, return_tensors='pt', padding='max_length',
                             truncation=True)

        model.config.min_length = min(len(inputs_1), len(inputs_2))

        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)

        alpha_steps = np.arange(0, 1.1, 0.1)

        for alpha in alpha_steps:
            encoder_outputs, target_len, (sentence_embed_1, sentence_embed_2) = model.get_noisy_embedding(
                input_ids_1=inputs_1["input_ids"],
                attention_mask_1=inputs_1["attention_mask"],
                input_ids_2=inputs_2["input_ids"],
                attention_mask_2=inputs_2["attention_mask"],
                alpha=alpha,
                sent_emb=True,
                mix_style=False)

            similarity = (sentence_embed_1 -sentence_embed_2).pow(2).sum(dim=2).sqrt().mean().detach().cpu().item()

            model_kwargs = {"encoder_outputs": encoder_outputs}

            sample_output = model.generate(input_ids=None, **model_kwargs, max_length=128)
            generated_sent = tokenizer.decode(sample_output[0], skip_special_tokens=True)

            reference_sent_1 = [tokenizer.convert_ids_to_tokens(inputs_1["input_ids"][0], skip_special_tokens=True)]
            reference_sent_2 = [tokenizer.convert_ids_to_tokens(inputs_2["input_ids"][0], skip_special_tokens=True)]

            candidate = tokenizer.convert_ids_to_tokens(sample_output[0], skip_special_tokens=True)

            bleu_sent_1 = sentence_bleu(reference_sent_1, candidate, weights=(1.0, 0.0, 0, 0))
            bleu_sent_2 = sentence_bleu(reference_sent_2, candidate, weights=(1.0, 0.0, 0, 0))

            fw.write("{}\t{}\t{}\t{}\t{}\n".format(alpha, generated_sent, bleu_sent_1, bleu_sent_2, similarity))

    fw.close()
    logger.info("Generation output saved at {}".format(gen_output_path))
    logger.info("**************************************")

def main():
    args = parse_args()

    args.gpu_ids = args.gpu_ids.split(",")
    args.gpu_ids = [int(i) for i in args.gpu_ids]

    # set output dir
    if args.do_train:
        args.output_dir = os.path.join(args.output_dir,
                                       "{}_bs_{}_gpus_{}_lr_{}_epochs_{}_{}_gnoise_{}_l2_{}_alpha_{}_inoise_{}".format(
                                           args.model_name_or_path.rsplit('/')[-1],
                                           args.per_device_train_batch_size,
                                           len(args.gpu_ids),
                                           args.learning_rate,
                                           args.num_train_epochs,
                                           args.train_mode,
                                           args.add_gaussian_noise,
                                        args.use_l2_loss,
                                       args.alpha_for_beta,
                                       args.input_noise_prob)
                                       )
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("output_dir is set to: {} (ckpt and eval result will be saved here)".format(args.output_dir))

        args.train_dataset_cached_dir = os.path.join(args.dataset_dir, "hf_cache/train_{}_cache".format(args.train_mode))
        os.makedirs(args.train_dataset_cached_dir, exist_ok=True)
        logger.info("train dataset will be cached at: {}".format(args.train_dataset_cached_dir))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in args.gpu_ids)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.gpu_ids)
    if args.do_train:
        logger.info("You are using {} gpus (gpu_ids: {})".format(args.n_gpu, args.gpu_ids))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if args.do_eval or args.do_generation or args.do_finetuning:
        config = AutoConfig.from_pretrained(args.my_checkpoint)
        model = BartForLINDA.from_pretrained(args.my_checkpoint,
                                             from_tf=bool(".ckpt" in args.model_name_or_path),
                                             config=config)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = BartForLINDA.from_pretrained(args.model_name_or_path,
                                             from_tf=bool(".ckpt" in args.model_name_or_path),
                                             config=config)
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # do train
    if args.do_train:
        train(args, tokenizer, model)

    # do eval
    if args.do_eval:
        evaluate(args, tokenizer, model)

    if args.do_generation:
        generation(args, tokenizer, model)


if __name__ == "__main__":
    main()
