# credit-risk-pipeline
Hi anh Kelvin, thank you for reviewing my project. Please see my answer for this test at ```DoMinhTam - Written Exercise - Data Science Lead - Credit Risks.pdf``` file.

## How to run the pipeline

### 1. Create a virtual environment
Because i use poetry to manage my project, so you need to install poetry first. By running:
``` make install-poetry ```

Then install all dependencies by running:
``` make install-deps ```

### 2. Run the pipeline
# You can swap between my two model in the pipeline by changing the model type in the config file
```
model:
  name: "default_of_credit_card_clients"
  version: "0.0.1"
  type: "gbm"  # Can be 'nn' for Neural Network or 'gbm' for Gradient Boosting Machine
  model_path: "models/${name}_${version}.pkl"
  predictions_path: "models/predictions.csv"
  hidden_layers: [64, 32, 16]
```
# To run the pipeline e2e, run:
``` make run-pipeline ```

# To run the pipeline with a specific step, run:
``` make <step-name> ```
Available steps: preprocess, train, evaluate, predict

# To clean up data and model, run:
``` make clean ```