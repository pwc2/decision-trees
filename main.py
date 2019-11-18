#!/usr/bin/env python3
"""
    File name: main.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/16/2019
    Python Version: 3.7

"""

import json
from pathlib import Path

import pandas as pd

from models.decision_tree import DecisionTree

train = pd.read_csv('data/pa3_train.csv')
validation = pd.read_csv('data/pa3_val.csv')
test = pd.read_csv('data/pa3_test.csv')

tree = DecisionTree(train, validation, test, label='class', max_depth=8)
results = tree.train()

# Save output for learned model to .json file.
output_folder = Path('model_output')
output_path = Path(__file__).parent.resolve().joinpath(output_folder)
training_file = output_path.joinpath(Path('tree.json'))

# Create output directory if doesn't exist.
if not Path(output_path).exists():
    Path(output_path).mkdir()
with open(training_file, 'w') as f:
    json.dump(results, f, indent=4)

# # Best validation accuracy with 14 iterations, calculate and save predictions.
# test_predictions = model.predict_test(learned_model['weights'][13])
# prediction_file = output_path.joinpath(Path('oplabel.csv'))
# with open(prediction_file, 'w') as fp:
#     writer = csv.writer(fp)
#     for i in test_predictions:
#         writer.writerows([[i]])
