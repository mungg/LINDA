import csv
import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single multiple choice question. Here "article" is optional"""
    def __init__(self, sent_1, sent_2):
        """Construct an instance."""
        self.sent_1 = sent_1
        self.sent_2 = sent_2

def create_examples(filepath):
    examples = []
    print('Loading sentence tokenized Wiki dump...')
    with open(filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        next(csv_reader)
        print('--- loading done ---')
        for id_, row in enumerate(csv_reader):
            sent_1, sent_2 = row
            examples.append(InputExample(
                sent_1=sent_1,
                sent_2=sent_2))
    return examples


def create_datasets(args, examples, tokenizer):
    max_target_length = args.max_target_length
    max_sequence = args.max_source_length

    features = []
    for example in examples:
        input_1, input_2 = example.sent_1, example.sent_2
        tokenized_sent_1 = tokenizer(input_1, max_length=args.max_source_length, padding=True, truncation=True)
        tokenized_sent_2 = tokenizer(input_2, max_length=args.max_source_length, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels_1 = tokenizer(input_1, max_length=args.max_target_length, padding=True, truncation=True)
            labels_2 = tokenizer(input_2, max_length=args.max_target_length, padding=True, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.

        padding = "max_length" if args.pad_to_max_length else False

        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels_1["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_1["input_ids"]
            ]
            labels_2["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_2["input_ids"]
            ]

        input_ids_1 = tokenized_sent_1.input_ids + [0] * (max_sequence - len(tokenized_sent_1.input_ids))
        attention_mask_1 = tokenized_sent_1.attention_mask + [0] * (max_sequence - len(tokenized_sent_1.attention_mask))
        input_ids_2 = tokenized_sent_2.input_ids + [0] * (max_sequence - len(tokenized_sent_2.input_ids))
        attention_mask_2 = tokenized_sent_2.attention_mask + [0] * (max_sequence - len(tokenized_sent_2.attention_mask))
        # todo: check if shifted
        labels_1 = labels_1.input_ids + [0] * (max_sequence - len(labels_1.input_ids))
        labels_2 = labels_2.input_ids + [0] * (max_sequence - len(labels_2.input_ids))

        features.append([
            input_ids_1,
            attention_mask_1,
            input_ids_2,
            attention_mask_2,
            labels_1,
            labels_2,
        ])

    all_input_ids_1 = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_attention_mask_2 = torch.tensor([f[3] for f in features], dtype=torch.long)
    all_labels_1 = torch.tensor([f[4] for f in features], dtype=torch.long)
    all_labels_2 = torch.tensor([f[5] for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids_1,
        all_attention_mask_1,
        all_input_ids_2,
        all_attention_mask_2,
        all_labels_1,
        all_labels_2,
    )
