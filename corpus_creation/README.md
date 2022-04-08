
# Corpora

This directory contains the scripts used to generate both the machine translated version of the text as well as the script for their automatic evaluation.


## `translate_Azure.py`
`translate_Azure.py` is the CLI script to call the Microsoft Translator API.  

Given an API key and a TSV file containing the original corpora, run the software from the terminal using:

```bash
py translate_Azure.py [-h] --filepath FILEPATH --key APIKEY --src SOURCE --trg TARGET
```

All other arguments are required and have the following behaviors:

`--filepath` The source text to be translated. This should be a PATH to a tsv file, containing two columns named as follows:


| src | trg |
| :-: | :-: |
| text | text |
| ... | ... |
| text | text |



`--src` The source language, formatted following the [ISO-639-1](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support)

`--trg` The target language, formatted following the [ISO-639-1](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/language-support)

`--key` The API key, saved in a separate TXT file.

The script generates a new TSV file, now containing the `MT` column and named `translated_{FILENAME}.tsv`







## `translate_MMT.py`
`translate_MMT.py` is the script to call the ModernMT API.
Given an API key and a TSV file containing the original corpora, run the software from the terminal using:

```bash
py translate_MMT.py [-h] --filepath FILEPATH --key APIKEY --src SOURCE --trg TARGET
```

All other arguments are required and have the following behaviors:

`--filepath` The source text to be translated. This should be a PATH to a tsv file, containing two columns named as follows:


| src | trg |
| :-: | :-: |
| text | text |
| ... | ... |
| text | text |



`--src` The source language, formatted following the [ISO-639-1](https://www.modernmt.com/api/#languages)

`--trg` The target language, formatted following the [ISO-639-1](https://www.modernmt.com/api/#languages)

`--key` The API key, saved in a separate TXT file.

The script generates a new TSV file, now containing the `MT` column and named `translated_{FILENAME}.tsv`




