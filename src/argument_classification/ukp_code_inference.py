"""
This code is taken from the following repository: 
https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering/blob/master/argument-classification/inference.py

Runs a pre-trained BERT model for argument classification.

You can download pre-trained models here: https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip

The model 'bert_output/ukp/bert-base-topic-sentence/all_ukp_data/' was trained on all eight topics (abortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms) from the Stab et al. corpus  (UKP Sentential Argument
Mining Corpus)

The code was adapted for usage in this repository
"""

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import pandas as pd

from config import cfg
from src.external_code.ukp_code_train import InputExample, convert_examples_to_features


num_labels = 3
model_path = 'models/argument_classification_ukp_all_data/'
label_list = ["NoArgument", "Argument_against", "Argument_for"]
max_seq_length = 64
eval_batch_size = 8


def create_input_examples(df, col):
    examples = []
    segments = list(df[col])
    topics = list(df["topic"])
    for seg, topic in zip(segments, topics):
        if str(seg) != "nan":
            examples.append(InputExample(text_a=topic, text_b=seg, label="NoArgument"))
    return examples


def predict(input_data):

    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    eval_features = convert_examples_to_features(input_data, label_list, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)
    model.eval()

    predicted_labels = []
    with torch.no_grad():
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()

            for prediction in np.argmax(logits, axis=1):
                predicted_labels.append(label_list[prediction])

    return predicted_labels


def save_argument_predictions(filepath, col, preds, col_class):
    df = pd.read_csv(filepath, encoding="iso-8859-1")
    preds_empty_lines = []
    idx = 0
    for id in list(df[col]):
        if str(id) == "nan":
            preds_empty_lines.append("")
        else:
            preds_empty_lines.append(preds[idx])
            idx += 1
    df[col_class] = preds_empty_lines
    df.to_csv(filepath, index=False)


'''
Main method for adding an arg_class column to an existing dataframe of segments
param filepath: the csv-file for which the argument classes should be added
param col: the column name with segments that should be classified
'''
def add_arg_class_to_file(filepath, col="segment"):
    col_class = "arg_class"
    df = pd.read_csv(filepath, encoding="iso-8859-1")
    examples = create_input_examples(df, col)
    preds = predict(examples)
    save_argument_predictions(filepath, col, preds, col_class)

