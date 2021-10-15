# Optimizing an ML Pipeline in Azure

## Overview
- This project is part of the Udacity Azure ML Nanodegree.
- In this project, we built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn model
- This model is then compared to an Azure AutoML run.

## Summary

#### Problem 

- Predict wether a client will subscribe to a particular term (binary classification)

#### Approach
- Employ to approaches: 
    - logistic regression utilizing hyperparameter tuning via AZ Hyperdrive
    - AZ AML

#### Result

- Both models performed similar in terms of accuracy
- One should, however, not fully rely on the model as the underlying data is imabalanced
- Further steps should address this issue, e.g. via SMOTE

## Scikit-learn Pipeline
- The scikit-learn pipeline was structured as follows:

    1. Collecting Data
    2. Cleaning the Data
    3. Splitting Data into Train & Test Samples
    4. Training and Tuning the Logistic Regression
    5. Saving the most accurate hyperparameter settings

## Parameter Sampling & Early Stopping Policy
- For performance reasons, only four runs were undertaken
- In principle, however, the advantages of the underlying strategy are as follows:

    - Random Sampling:
        - computationally less demanding than Grid Sampling or a Bayesian Sampler
        - while beforementioned options might prove more accurate, Random Sampling serves as a well suited entrypoint
    
    - Bandit Policy:
        - early stoppage of training iterations in case of worsening of accuracy as compared to best performing training model
        - for the approach at hand, however, the SDK seemed to have some issues incorporating the policy 

## AutoML
- AZ AML provides the user with a possibility to tune a multitude of models for classification, regression and time series scenarios.
- It is particularly well suited to check for promising approaches to a certain problem

## Pipeline comparison
- With some surprise, the automated model did not perform significantly better than the simple  hypertuned approach
- Quantitavely, the logistic regression returned an accuracy of .9112 on test data, whereas the best performing AZ AML approach of using a Voting Ensemble of algorithms proved to be .9175 accurate

## Future work
- AZ prompted a warning in relation to class imbalance
- As such, to improve classification for out of sample observations, SMOTE or some other method of handling class imbalance seems promising
- It might be worth to check out further metrics to ensure reliable out of sample classifications

