"""
    File name: run_part2.py
    Author: Patrick Cummings
    Date created: 11/17/2019
    Date last modified: 11/19/2019
    Python Version: 3.7

"""
import json
import random
from pathlib import Path

import pandas as pd

from models.random_forest import RandomForest

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

# Vary tree size.
for n in [1, 2, 5, 10, 25]:
    # Create random forest with n trees, depth = 2, and n_features = 5 and save results in model_output.
    rf = RandomForest(train_set, validation_set, test_set, label='class', n_trees=n, n_features=5, seed=1, max_depth=2)
    results = rf.train()

    # Save output for learned model to .json file.
    output_folder = Path('model_output/part2')
    output_path = Path(__file__).parent.resolve().joinpath(output_folder)
    training_file = output_path.joinpath(Path('rf_ntrees_' + str(n) + '_nfeat_' + str(rf.m) + '.json'))

    # Create output directory if doesn't exist.
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(training_file, 'w') as f:
        json.dump(results, f, indent=4)

# Vary number of bagged features.
for m in [1, 2, 5, 10, 25, 50]:
    # Create random forest with 15 trees, depth = 2, and n_features = m and save results in /model_output.
    rf = RandomForest(train_set, validation_set, test_set, label='class', n_trees=15, n_features=m, seed=1, max_depth=2)
    results = rf.train()

    # Save output for learned model to .json file.
    output_folder = Path('model_output/part2')
    output_path = Path(__file__).parent.resolve().joinpath(output_folder)
    training_file = output_path.joinpath(Path('rf_ntrees_15' + '_nfeat_' + str(rf.m) + '.json'))

    # Create output directory if doesn't exist.
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(training_file, 'w') as f:
        json.dump(results, f, indent=4)

# Vary random seed with best parameters from models created above.
n = 15
m = 25
i = 1  # index to include in name of saved files
random.seed(0)
seeds = random.sample([x for x in range(10000)], k=10)
for s in seeds:
    # Create random forest with 15 trees, depth = 2, and n_features = 25 and save results in /model_output.
    rf = RandomForest(train_set, validation_set, test_set, label='class', n_trees=n, n_features=m, seed=s, max_depth=2)
    results = rf.train()

    # Save output for learned model to .json file.
    output_folder = Path('model_output/part2')
    output_path = Path(__file__).parent.resolve().joinpath(output_folder)
    training_file = output_path.joinpath(Path('rf_ntrees_15' + '_nfeat_' + str(rf.m) + '_seed_' + str(i) + '.json'))

    # Create output directory if doesn't exist.
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    with open(training_file, 'w') as f:
        json.dump(results, f, indent=4)
    i += 1
