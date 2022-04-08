# MTsweet

This repository contains the scripts used in the experiments of my dissertation for the master's in Specialized Translation at the University of Bologna.

It represents an attempt at approaching the problem of assessing machine translation quality starting from the source text alone. The work involved first building a corpus pairing source segments with the evaluation scores obtained by their respective machine translated version using state-of-the-art metrics for MT Evaluation ([cushLEPOR](https://github.com/poethan/cushLEPOR), [BERTScore](https://github.com/Tiiiger/bert_score)) and MT Quality Estimation ([COMET](https://github.com/Unbabel/COMET), [TransQuest](https://github.com/TharinduDR/TransQuest)).
On the basis of that corpus, the multilingual model XLM-RoBERTa (base) was fine-tuned and evaluated to predict those same scores, once as a single-task model and once as a multi-task model.

If you wish to know more about this research, please refer to the full dissertation:

> Return to the Source: Assessing Machine Translation Suitability based on the Source Text using XLM-RoBERTa


## Development setup

All required dependencies are listed in the `Pipfile`. You can directly install a virtual environment using pipenv and running the following command:

```sh
pipenv install Pipfile
```


## Subfolders

`corpus_creation` contains the scripts used to generate both the machine translated version of the text as well as the script for their automatic evaluation.

`models` contains the scripts used to fine-tune XLM-RoBERTa, once as a single-task model and once as a multi-task model.


## Meta

[Francesco Fernicola](https://www.linkedin.com/in/francesco-fernicola-69a0771b7/?locale=en_US) – [@FrancescoDaimon](https://twitter.com/FrancescoDaimon) – daimon.f@outlook.com

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


