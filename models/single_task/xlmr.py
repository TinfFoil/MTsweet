from datasets import load_dataset, Dataset, DatasetDict, load_metric
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, get_scheduler, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
from torch.utils.data import DataLoader
import torch 
import datasets
from torch.nn import MSELoss
from transformers import TrainingArguments
import wandb
from scipy import stats
from torch import nn
from transformers import Trainer
import argparse
import os

def main(metric_idx, lr, num_train_epochs, batch_size, output_dir):
    
    
    # Below is the dataset I have used and the split is made according to the formatting explained in the paper.
    
    train = pd.read_csv('corpora/train.tsv', sep='\t', header=0)
    validate = pd.read_csv('corpora/test.tsv', sep='\t', header=0)

    train['scores'] = train['scores'].map(lambda x: json.loads(x)[metric_idx])
    validate['scores'] = validate['scores'].map(lambda x: json.loads(x)[metric_idx])

    train = train.rename(columns={"en": "sentence", "scores": "label"})
    validate = validate.rename(columns={"en": "sentence", "scores": "label"})


    # Format dataset for Huggingface
    train_dataset = Dataset.from_pandas(train)
    validate_dataset = Dataset.from_pandas(validate)

    dict_data = {"train": train_dataset, "validation": validate_dataset}


    raw_datasets = DatasetDict(dict_data)
    checkpoint = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(dataset):
        return tokenizer(dataset['sentence'], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    task_dict = {"hyp":[], "gold":[]}

    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", learning_rate=lr, num_train_epochs=num_train_epochs, per_device_train_batch_size=batch_size, save_strategy="no")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    wandb.init(mode="disabled")

    trainer = HuberTrainer(
                    model,
                    training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    )
    
    trainer.train()

    final_output = pd.DataFrame.from_dict(task_dict)
    final_output.to_csv(f"{output_dir}/{num_train_epochs}e_{batch_size}b_{metric_idx}m.tsv", sep="\t", index_label="idx")

def compute_metrics(eval_preds):
    metric = load_metric("rmse")
    logits, labels = eval_preds
    predictions = logits.flatten()
    task_dict["hyp"].extend(list(predictions))
    task_dict["gold"].extend(list(labels))
    return metric.compute(predictions=predictions, references=labels)


class HuberTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.HuberLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, type=int, metavar='METRIC_IDX', help='Index of the metric on which the model will be trained: cushLEPOR = 0, BERTScore = 1, COMET = 2, TransQuest = 3')
    parser.add_argument('--lr', required=True, type=float, metavar='LEARNING_RATE', help='Learning rate for the model')
    parser.add_argument('--num_train_epochs', required=True, type=int, metavar='TRAIN_EPOCHS', help='Number of epochs to train the model')
    parser.add_argument('--batch_size', required=True, type=int, metavar='DEVICE_BATCH_SIZE', help='Size of the batch to load on each device')
    parser.add_argument('--save_to', type=str, nargs='?', default=os.getcwd(), metavar='SAVE_PATH', help='(optional) Path where the output will be saved')
    
    args = parser.parse_args()
    print(args)
    print(parser)
    main(args.metric, args.lr, args.num_train_epochs, args.batch_size, args.save_to)
