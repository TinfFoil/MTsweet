from multitask_data_collator import DataLoaderWithTaskname
import nlp
import numpy as np
import torch
import transformers
from datasets import load_metric
import pandas as pd

def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    output_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    for idx, task_name in enumerate(["BERTScore", "CushLEPOR", "COMET", "TransQuest"]):
        val_len = len(features_dict[task_name]["validation"])
        task_dict = {"text":[], "hyp":[], "gold":[]}
        metric = load_metric("rmse")
        for index in range(0, val_len, batch_size):

            batch = features_dict[task_name]["validation"][index : min(index + batch_size, val_len)]["doc"]
            
            labels = features_dict[task_name]["validation"][index : min(index + batch_size, val_len)]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True)

            inputs["input_ids"] = torch.LongTensor(inputs["input_ids"]).cuda()
            inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"]).cuda()
            
            logits = multitask_model(task_name, **inputs).logits
            print("logits", logits)
            """
            predictions = torch.argmax(torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),dim=1)           
            """
            predictions = logits.detach().cpu().flatten().tolist()
            print("predictions", predictions)
            print("labels", labels)
            task_dict["hyp"].extend(predictions)
            task_dict["text"].extend(list(batch))
            task_dict["gold"].extend(list(labels))
            metric.add_batch(predictions=predictions, references=labels)     
        eval = metric.compute()
        output_dict[task_name] = task_dict
        preds_dict[task_name] = [eval["rmse"]]
        print(f"\n\nTask name: {task_name}\t{eval}\n\n")
    
    preds = pd.DataFrame.from_dict(preds_dict)
    for task, output in output_dict.items():    
        final_output = pd.DataFrame.from_dict(output)
        final_output.to_csv(f"10epoch_{task}.tsv", sep="\t", index_label="idx")
    
    
    print(preds)
