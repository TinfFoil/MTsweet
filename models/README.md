
# Models

This directory contains the scripts used to fine-tune the XLM-RoBERTa (base) implementation by HuggingFace, once as a single-task model and once as a multi-task model.
For more details regarding the exact characteristics and performance of each model, please refer to Chapter 3 and Chapter 4 of the thesis.


## Single-task model
`xlmr.py` is the CLI script used to train a simple XLM-R model.

```bash
pipenv run python xlmr.py [-h] --metric METRIC_IDX --lr LEARNING_RATE --num_train_epochs TRAIN_EPOCHS --batch_size DEVICE_BATCH_SIZE [--save_to SAVE_PATH]
```

The following arguments are required and have the following behaviors:

`--metric` The single metric on which the model will be trained. It is based off of its index from the original TSV file and requires an integer as input. Following are the values for each metric: cushLEPOR = 0, BERTScore = 1, COMET = 2, TransQuest = 3

`--lr` Learning rate for the model. For my experiments I used a learning rate of 2e-5.

`--num_train_epochs` Number of epochs for model training.

`--batch_size` Size of the batch to load on each device.

Additionally, an optional argument can be provided if one wants to save the output to a particular folder, namely:

`--save_to` Path where the output will be saved. By default, this is the current working directory.


## Multi-task model
A multi-task model using transformers normally works by having a shared encoder transformer, with different task heads for each task. In this case, I have instead adopted a Shared Encoder approach, where multiple encoders are loaded and mapped, in order to share their parametres. Below is a visualization of the model: <br>

![multitask](https://user-images.githubusercontent.com/56536141/164017192-4d703230-7f62-4e4b-9a08-4adb1394fa28.PNG)


`main.py` is the CLI script used to train a multi-task XLM-R model. In order to visualize the full list of optional arguments, please run

```bash
pipenv run python main.py [-h] 
```

If you wish to replicate the results, here is an example of the command used to run the experiments explained in Chapter 4:

```bash
pipenv run python main.py --model_name_or_path='xlm-roberta-base' --per_device_train_batch_size=1 --output_dir=output --num_train_epochs=5 --learning_rate=2e-5
```


These experiments are adapted from the experiments used as a presentation for`jiant`. You can consult the original link here: <br>
[Multi-task Training with Transformers+NLP](https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=CQ39AbTAPAUi)
