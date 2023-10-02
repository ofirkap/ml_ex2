# Machine Learning Assignment - Kaggle-style Competition

## Overview
This assignment simulates a Kaggle-style competition with a dataset divided into train, dev, and test sets. The task is to create an accurate classifier, with the top classifiers receiving bonus points. The assignment consists of two main parts:

## Part 1: Experimental Analysis (`experiments.ipynb`)
### Objective
- Conduct preliminary data analysis to build and justify the selected model.

### Tasks
- Exploration and preliminary data analysis.
- Justification for model selection.
- Demonstrate preprocessing steps and their impact on model accuracy.
- Describe hyperparameter search and its impact on model accuracy.

## Part 2: Model Development (`model.py`)
### Objective
- Build and submit a model using the provided template.

### Tasks
- Adjust the template according to the chosen model from Part 1.
- Set base hyperparameters for the model.
- Allowed to add functions but must be confined to a single file submission.

## Solution Overview
- The solution begins with importing necessary libraries and loading the dataset.
- Preliminary data analysis reveals an unbalanced dataset.
- LazyPredict is utilized to select the best algorithms for further adjustmemt.
- The dataset is balanced using over-sampling and under-sampling.
- Feature selection and data transformation techniques are applied.
- Various models are tested, compared, and adjusted using different parameters and features.
- Final models are validated against the 'dev' set to evaluate their performance.

## File Descriptions
- `experiments.ipynb`: Jupyter notebook containing experimental analysis including data exploration, model selection, and hyperparameter search.
- `model.py`: Python script with the final model implementation, ready for submission.

## How to Run
1. Load and run `experiments.ipynb` for experimental analysis and model selection.
2. Run `model.py` to train and test the final model using selected parameters from the experiment.

## Dependencies
- Numpy
- Pandas
- scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- seaborn
- LazyPredict

## Conclusion
This assignment provides hands-on experience in Kaggle-style competitions, involving classifier selection, data preprocessing, model development, and hyperparameter search.
