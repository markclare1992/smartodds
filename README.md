# Football Modelling
This repository contains the code for modelling football outcomes.

## Installation
To install the package (using poetry):
First clone the repository:
```bash
git clone {repo}
```
Create (optional) or activate a virtual environment:
```bash
poetry shell
```
or using virtualenv:
```bash
pyenv virtualenv 3.10.12 football_modelling
pyenv activate football_modelling
```

Install the package:
```bash
poetry install
```

# Usage
Example usage below:
```python
from smartodds.model.extended_model import ExtendedModel
extended_model = ExtendedModel('merged_data.csv','data_mls_simset_predictions.csv')
extended_model.fit()
extended_model.add_predictions()
extended_model.display_fitted_model_params()
extended_eval=extended_model.evaluate()
comparison_eval=extended_model.evaluate_comparison_data()
extended_model.add_expected_points()
expected_points_table = extended_model.create_expected_points_table()
```

Inspect fitted rates/probs etc:
```python
test_df= extended_model.test
test_df.head()
```
