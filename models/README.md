
# Models

This directory contains the scripts used to fine-tune the XLM-RoBERTa (base) implementation by HuggingFace, once as a single-task model and once as a multi-task model.
For more details regarding the exact characteristics and performance of each model, please refer to Chapter 3 and Chapter 4 of the thesis.


## Single-task model
`xlmr.py` is the CLI script used to train a simple XLM-R model.

```bash
py xlmr.py [-h] --metric METRIC_IDX --lr LEARNING_RATE --num_train_epochs TRAIN_EPOCHS --batch_size DEVICE_BATCH_SIZE [--save_to SAVE_PATH]
```

The following arguments are required and have the following behaviors:

`--metric` The single metric on which the model will be trained. It is based off of its index from the original TSV file and requires an integer as input. Following are the values for each metric: cushLEPOR = 0, BERTScore = 1, COMET = 2, TransQuest = 3

`--lr` Learning rate for the model. For my experiments I used a learning rate of 2e-5.

`--num_train_epochs` Number of epochs for model training.

`--batch_size` Size of the batch to load on each device.

Additionally, an optional argument can be provided if one wants to save the output to a particular folder, namely:

`--save_to` Path where the output will be saved. By default, this is the current working directory.


## Multi-task model
A multi-task model using transformers normally works by having a shared encoder transformer, with different task heads for each task. In this case, I have instead adopted a Shared Encoder approach, where multiple encoders are loaded and mapped, in order to share their parametres. The final result can be visualized as follows: 

![multitask](https://user-images.githubusercontent.com/56536141/164017192-4d703230-7f62-4e4b-9a08-4adb1394fa28.PNG)

`main.py` is the CLI script used to train a simple XLM-R model.

```bash
py main.py [-h] --metric METRIC_IDX --lr LEARNING_RATE --num_train_epochs TRAIN_EPOCHS --batch_size DEVICE_BATCH_SIZE [--save_to SAVE_PATH]
```



References:
[Multi-task Training with Transformers+NLP](https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=CQ39AbTAPAUi)
