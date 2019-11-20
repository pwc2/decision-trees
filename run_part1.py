"""
    File name: run_part1.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/19/2019
    Python Version: 3.7

"""
import json
from pathlib import Path

import pandas as pd

from models.decision_tree import DecisionTree

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

# Create trees with depths from 1 to 8 inclusive and save results in /model_output.
for depth in range(1, 9):
    tree = DecisionTree(train_set, validation_set, test_set, label='class', max_depth=depth)
    results = tree.train()

    # Save output for learned model to .json file.
    output_folder = Path('model_output/part1')
    output_path = Path(__file__).parent.resolve().joinpath(output_folder)
    training_file = output_path.joinpath(Path('tree' + str(depth) + '.json'))

    # Create output directory if doesn't exist.
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(training_file, 'w') as f:
        json.dump(results, f, indent=4)
