import json
import datasets
from random import randint
from nlp import DatasetInfo, BuilderConfig, SplitGenerator, Split, utils
import pandas as pd


logger = datasets.logging.get_logger(__name__)


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for MultitaskDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)


class MultitaskDataset(datasets.GeneratorBasedBuilder):
    """MultitaskDataset: Version 1.0.0"""

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Multitask dataset",
            features=datasets.Features(
                {
                    "doc": datasets.Value("string"),
                    "target": datasets.Value("float32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        data = None
        if isinstance(filepath, list):
            filepath = filepath[0]
        if ".tsv" in filepath:
            data = pd.read_csv(filepath, sep="\t")
        else:
            data = pd.read_csv(filepath)
        data = data
        for idx, row in data.iterrows():
            yield row["idx"], {
                "doc": row["sentence"],
                "target": row["label"],
            }
