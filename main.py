from datasets import load_metric, load_dataset, ClassLabel, Sequence
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import argparse
import os
import json

#-----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='RISE Assignment')
parser.add_argument('--exp', type=str, default='A',
                    help='Experiment A with full taglist, Experiment B with reduced taglist.')

args = parser.parse_args()
experimentType = args.exp

#-----------------------------------------------------------------------------------------------------------------------
# Check for PyTorch
#-----------------------------------------------------------------------------------------------------------------------

import torch
if torch.cuda.is_available():
    print("torch CUDA is available")

#-----------------------------------------------------------------------------------------------------------------------
# Using the XLNet LLM model for fine-tuning on the MultiNERD Named Entity Recognition Dataset
#-----------------------------------------------------------------------------------------------------------------------

model_checkpoint = "xlnet-base-cased"
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#-----------------------------------------------------------------------------------------------------------------------
# Defining full taglist and subset of the taglist (PER, ORG, LOC, ANIM, and DIS)
#-----------------------------------------------------------------------------------------------------------------------

label_list = {}

all_tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
}

reduced_tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-DIS": 13,
    "I-DIS": 14,
}

if experimentType=="A":
    label_list = all_tags
elif experimentType=="B":
    label_list = reduced_tags

label_id_list = {v: k for k, v in label_list.items()}
label_id_list = dict(sorted(label_id_list.items()))

#-----------------------------------------------------------------------------------------------------------------------
# Defining the functions to tokenize and load the MultiNERD dataset
#-----------------------------------------------------------------------------------------------------------------------

def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def remove_excess_tags(example, filtered_tags):
    ner_tags = example["ner_tags"]
    example["ner_tags"] = [0 if tag not in filtered_tags else tag for tag in ner_tags]
    return example

def labels_not_sequential(label_dict):
    expected_key = 0
    change_ids = {}
    for key, value in label_dict.items():
        if key-expected_key>0:
            change_ids[key] = expected_key
        expected_key += 1
    return dict(sorted(change_ids.items(), key=lambda x: x[0], reverse=True))

def change_label_main(label_list, labels_to_change):
    new_label_list = {}
    for _label, _id in label_list.items():
        if _id in labels_to_change:
            new_label_list[_label] = labels_to_change[_id]
        else:
            new_label_list[_label] = _id
    label_id_list = {v: k for k, v in new_label_list.items()}
    return new_label_list, label_id_list

def change_labels(example, labels_to_change):
    ner_tags = example["ner_tags"]
    modified_ner_tags = []
    to_swap = list(labels_to_change.keys())
    for i in ner_tags:
        if i in to_swap:
            label_id = labels_to_change[i]
        else:
            label_id = i
        modified_ner_tags.append(label_id)
    example["ner_tags"] = modified_ner_tags
    return example

def replace_id_with_labels(example, mapping_dict):
    ner_tags = example["ner_tags"]
    modified_ner_tags = []
    for i in ner_tags:
        modified_ner_tags.append(mapping_dict.get(i,i))
    example["ner_tags"] = modified_ner_tags
    return example

#-----------------------------------------------------------------------------------------------------------------------
# Codes to load the MultiNERD dataset and tokenize
#-----------------------------------------------------------------------------------------------------------------------

dataset = load_dataset("Babelscape/multinerd", split=None)
dataset_split = tuple(dataset.keys())

for split in dataset_split:
    dataset[split] = dataset[split].filter(lambda data: data["lang"] == "en")
    dataset[split] = dataset[split].remove_columns("lang")

if experimentType=="B":
    for split in dataset_split:
    	dataset[split] = dataset[split].map(remove_excess_tags, fn_kwargs={"filtered_tags": label_list.values()}, num_proc=4)

labels_to_change = labels_not_sequential(label_id_list)
if labels_to_change:
    label_list, label_id_list = change_label_main(label_list, labels_to_change=labels_to_change)

    for split in dataset_split:
        dataset[split] = dataset[split].map(change_labels, fn_kwargs={"labels_to_change": labels_to_change}, num_proc=4)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

train_tokenized_datasets = train_tokenized_datasets.map(replace_id_with_labels, fn_kwargs={"mapping_dict": label_id_list}, num_proc=4)
test_tokenized_datasets = test_tokenized_datasets.map(replace_id_with_labels, fn_kwargs={"mapping_dict": label_id_list}, num_proc=4)

#-----------------------------------------------------------------------------------------------------------------------
# Defining the model and Training arguments
#-----------------------------------------------------------------------------------------------------------------------

label_list = list(label_list.keys())
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"rise-ner-{model_checkpoint}-{experimentType}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.00001,
    save_steps=100000,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

#-----------------------------------------------------------------------------------------------------------------------
# Computing the metrics
#-----------------------------------------------------------------------------------------------------------------------

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    for i in results.keys():
        if not i.startswith("overall"):
            print(f"Entity: {i}")
            for res in results[i]:
                if res != "number":
                    print(f"{res} \t {results[i][res]}")

    metrics_fname = f"metrics_of_{model_checkpoint}_{experimentType}.jsonl"
    if not os.path.exists(metrics_fname):
        open(metrics_fname, "w").close()

    with open(metrics_fname, "a") as f:
        f.write(json.dumps(results, default=str) + "\n")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#-----------------------------------------------------------------------------------------------------------------------
# Start Training
#-----------------------------------------------------------------------------------------------------------------------

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

trainer.save_model(f'rise_{model_checkpoint}_{experimentType}.model')
