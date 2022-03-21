import transformers


def convert_to_features(example_batch, model_name="xlm-roberta-base", max_length=512):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    inputs = list(example_batch["doc"])

    features = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = example_batch["target"]
    return features
