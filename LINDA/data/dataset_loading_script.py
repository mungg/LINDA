# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets

# TODO: dataset path
data_dir = "./LINDA_ICML/LINDA/data/raw"
TRAIN_DATA_PATH = os.path.join(data_dir, "train_tiny.txt") # to use the tiny version, change it to "train_tiny.txt"
DEV_DATA_PATH = os.path.join(data_dir, "dev.txt")


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""


# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

class MyWikiConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(MyWikiConfig, self).__init__(**kwargs)


class MyWikiDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MyWikiConfig(
            name="mywiki",
            version=datasets.Version("1.0.0"),
            description="My Wikipedia Dataset: sentence tokenized",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "sent_1": datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="HOMEPAGE",
            # License for the dataset if available
            license="LICENSE",
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": TRAIN_DATA_PATH}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": DEV_DATA_PATH}
            )
        ]

    def _generate_examples(
        self, filepath):
        """ Yields examples as (key, example) tuples. """
        with open(filepath, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                # if idx > 100:
                #     break
                yield idx, {"sent_1": row.strip('\n')}
        #     source_sentences = f.read().split("\n")
        #
        # random.shuffle(source_sentences)
        #
        # for id_ in range(100000000):
        #     yield id_, {"sent_1": source_sentences[id_]}
