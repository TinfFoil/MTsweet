import transformers
import torch


def save_model(model_name, multitask_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in ["BERTScore", "CushLEPOR", "COMET", "TransQuest"]:
        multitask_model.taskmodels_dict[task_name].save_pretrained(f"./final_models/{task_name}_model/")
