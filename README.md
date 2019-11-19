# decision-trees

`decision-trees` contains implementations of the decision tree classifier, random forest, and boosted trees (with AdaBoost) for binary categorical data.

### Requirements:

- `numpy 1.17.3`

- `pandas 0.25.2`

### Usage:

#### For a single decision tree:

```python
import pandas as pd

from models.decision_tree import DecisionTree

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

tree = DecisionTree(train=train_set, validation=validation_set,test=test_set, label='class', max_depth=2)
results = tree.train()
```

#### For a random forest:

```python
import pandas as pd

from models.random_forest import RandomForest

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

rf = RandomForest(train=train_set, validation=validation_set, test=test_set, label='class', n_trees=5, 
                    n_features=5, seed=1, max_depth=2)
results = rf.train()
```

#### For a boosted trees with AdaBoost:

```python
import pandas as pd

from models.boosted_trees import BoostedTrees

train_set = pd.read_csv('data/pa3_train.csv')
validation_set = pd.read_csv('data/pa3_val.csv')
test_set = pd.read_csv('data/pa3_test.csv')

boosted_trees = BoostedTrees(train=train_set, validation=validation_set, test=test_set, label='class', n_classifiers=5, max_depth=2)
results = boosted_trees.train()
```

### Data:

The `data/` folder contains .csv files with training, validation, and test sets.

### To run models:

- `run_part1.py` creates decision trees with varied depths.
- `run_part2.py` creates random forests with varied parameters.
- `run_part3.py` creates boosted trees with varied parameters.

`python main.py` will run all three parts in order, output will be saved in `model_output` folder.