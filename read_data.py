import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data

def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased', train_aug=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)
    print(data_path+r'\train.csv')
    train_df = pd.read_csv(data_path+r'\train.csv', header=None, encoding='ISO-8859-1')
    test_df = pd.read_csv(data_path+r'\test.csv', header=None, encoding='ISO-8859-1')
    valid_df = pd.read_csv(data_path+r'\valid.csv', header=None, encoding='ISO-8859-1')
    #unlabeled_df = pd.read_csv(data_path+'unlabeled.csv', header=None, encoding='ISO-8859-1')

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([v-1 for v in train_df[1]])
    train_text = np.array([v for v in train_df[0]])

    test_labels = np.array([u-1 for u in test_df[1]])
    test_text = np.array([v for v in test_df[0]])

    val_labels = np.array([u-1 for u in valid_df[1]])
    val_text = np.array([v for v in valid_df[0]])

    #unlabeled_text = np.array([v for v in unlabeled_df[0]])

    n_labels = max(train_labels) + 1
    print(n_labels)
    # Split the labeled training set, unlabeled training set, development set
    # train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
    #     train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)
    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text, train_labels, tokenizer, max_seq_len, train_aug)
    #train_unlabeled_dataset = loader_unlabeled(        unlabeled_text, tokenizer, max_seq_len, Translator(data_path))

    val_dataset = loader_labeled(
        val_text, val_labels, tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)

    return train_labeled_dataset, val_dataset, test_dataset, n_labels
    #return train_unlabeled_dataset


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        # self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.aug is not None:
            u, v, ori = (self.text[idx], self.text[idx], self.text[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)
