import glob
import math
import os
import random
import re
import csv

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed relevant RNGs.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_val_loss(checkpoint_path: str) -> float:
    """Extract validation loss from checkpoint path.

    E.g. checkpoint path format: path_to_dir/checkpoint_epoch=4-val_loss=0.450662.ckpt

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Parsed validation loss, if available.
    """
    match = re.search("val_loss=(.+?).ckpt", checkpoint_path)
    if match:
        return float(match.group(1))
    else:
        raise ValueError


def extract_step_or_epoch(checkpoint_path: str) -> int:
    """Extract step or epoch number from checkpoint path.

    E.g. checkpoint path formats:
        - path_to_dir/checkpoint_epoch=4.ckpt
        - path_to_dir/checkpoint_epoch=4-step=50.ckpt

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Parsed step or epoch number, if available.
    """
    if "step" in checkpoint_path:
        regex = "step=(.+?).ckpt"
    else:
        regex = "epoch=(.+?).ckpt"

    match = re.search(regex, checkpoint_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError


def get_best_checkpoint(checkpoint_dir: str) -> str:
    """Get best checkpoint in directory.

    Args:
        checkpoint_dir: Directory of checkpoints.

    Returns:
        Path to best checkpoint.
    """
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.ckpt"))

    try:
        # Get the checkpoint with lowest validation loss
        sorted_list = sorted(checkpoint_list, key=lambda x: extract_val_loss(x.split("/")[-1]))
    except ValueError:
        # If validation loss is not present,
        # get the checkpoint with highest step number or epoch number
        sorted_list = sorted(
            checkpoint_list, key=lambda x: extract_step_or_epoch(x.split("/")[-1]), reverse=True
        )

    return sorted_list[0]

####################################################################### for LINDA data preprocessing ###################

def extract_pairs(wiki_txt, train_output_path, dev_output_path):
    # WIKI_TXT = "data/wikipedia/enwiki_sentences_filtered.txt" # each line is a sentence
    # WIKI_CSV = "data/wikipedia/enwiki_sentences_filtered.csv" # columns: sent_1, sent_2
    print('Loading sentence tokenized Wiki dump...')
    f = open(wiki_txt, "r", encoding="utf-8")
    data = f.read()
    print('--- loading done ---')
    data = data.split('\n')
    random.shuffle(data)
    train_data = data[:round(len(data)*0.95)]
    dev_data = data[round(len(data)*0.95): ]

    print('--- sampling train pairs ---')
    train_pairs = [random.sample(train_data, 2) for _ in range(100000000)]
    with open(train_output_path, mode="w", encoding='utf-8') as my_file:
        fieldnames = ['sent_1', 'sent_2']
        writer = csv.DictWriter(my_file, fieldnames=fieldnames)
        writer.writeheader()
        for n, ex in enumerate(train_pairs):
            writer.writerow({'sent_1': ex[0], 'sent_2': ex[1]})
            if n % 10000000 == 0:
                print('{}% processed'.format(n/100000000 * 100))
        print(train_output_path, "saved")

    print('--- sampling dev pairs ---')
    dev_pairs = [random.sample(dev_data, 2) for _ in range(10000)]
    with open(train_output_path, mode="w", encoding='utf-8') as my_file:
        fieldnames = ['sent_1', 'sent_2']
        writer = csv.DictWriter(my_file, fieldnames=fieldnames)
        writer.writeheader()
        for ex in dev_pairs:
            writer.writerow({'sent_1': ex[0], 'sent_2': ex[1]})
        print(dev_output_path, "saved")

def mask_tokens(inputs, tokenizer, noise_prob):
    # this code from alps/
    labels = inputs.clone()

    if noise_prob == 0.0:
        return inputs
    probability_matrix = torch.full(inputs.shape, noise_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return inputs

def mask_words(sents, mask_prob, tokenizer, is_span=True):

    if mask_prob == 0.0:
        return sents
    masked_sents = []
    mask_token = tokenizer.mask_token
    for sent in sents:
        word_list = np.array(sent.strip().split(" "))
        word_num = len(word_list)
        num_to_mask = int(math.ceil(word_num * mask_prob))
        indices = np.random.permutation(np.arange(word_num))
        word_list[indices[:num_to_mask]] = mask_token

        if is_span:
            # replace multi <mask> to single one
            span_masked_word_list = []
            prev = ''
            for word in word_list:
                if word == mask_token and prev == mask_token:
                    pass
                else:
                    span_masked_word_list.append(word)
                prev = word

            masked_sent = " ".join(span_masked_word_list)
        else:
            masked_sent = " ".join(word_list)
        masked_sents.append(masked_sent)
    return masked_sents
