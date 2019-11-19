"""
    File name: run_part3.py
    Author: Patrick Cummings
    Date created: 11/18/2019
    Date last modified: 11/18/2019
    Python Version: 3.7

"""

import json
from pathlib import Path

import pandas as pd

from models.boosted_trees import BoostedTrees

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

# Run AdaBoost with varied number of base classifiers (L)
for L in [1, 2, 5, 10, 15]:
# for L in [5]:
    boosted_trees = BoostedTrees(train_set, validation_set, test_set, label='class', n_classifiers=L, max_depth=1)
    results = boosted_trees.train()

    # Save output for learned model to .json file.
    output_folder = Path('model_output/part3')
    output_path = Path(__file__).parent.resolve().joinpath(output_folder)
    training_file = output_path.joinpath(Path('boosted_L_' + str(L) + '.json'))

    # Create output directory if doesn't exist.
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(training_file, 'w') as f:
        json.dump(results, f, indent=4)

# Run AdaBoost with L = 6 and max depth = 2
L = 6
boosted_trees = BoostedTrees(train_set, validation_set, test_set, label='class', n_classifiers=L, max_depth=2)
results = boosted_trees.train()

# Save output for learned model to .json file.
output_folder = Path('model_output/part3')
output_path = Path(__file__).parent.resolve().joinpath(output_folder)
training_file = output_path.joinpath(Path('boosted_L_' + str(L) + '_depth_2' + '.json'))

# Create output directory if doesn't exist.
if not Path(output_path).exists():
    Path(output_path).mkdir()
with open(training_file, 'w') as f:
    json.dump(results, f, indent=4)