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
- The AML settings where structured as follows:
    - primary metric: accuracy, to allow comparision with the hypertuned model
    - max concurrent iterations: 4, respecting the modest computing ressources
    - experiment timeout: 30, see the above
    - enable early stopping: True, to stop computing new iterations if the chosen metric calculated per iteration seems to have converged
    - cross validations folds: 4, 25% of the data is used in each of the 4 validation folds per iteration
    - verbosity: logging.Info, amount of info logged
- The aforementioned AML settings where subsequentely used in the AML config. The config furthermore specified the following:
    - task: classification (binary classification task)
    - cores per iteration: -1, the maximum number of available cores
    - compute target: cpu_cluster, the provided compute
    - training_data: data, the full dataset analyzed using crossvalidation
    - label column: y, the variable of interest in the dataset

## Pipeline comparison
- With some surprise, the automated model did not perform significantly better than the simple  hypertuned approach
- Comparing the two approaches, this is remarkable given that the automated approach applies a host of models including some preprocessing and feature engineering. On the other hand, the hypertuned model only applies a GLM relying on a logistic distribution - albeit constrained. Personally, I'd expected for the AML to perform significantly better than the baseline-logit as it utilizes more advanced techniques based on Gradient Boosting, Tree Algorithms and Ensembling.
- Quantitavely speaking, the logistic regression returned an accuracy of .9112 on test data, whereas the best performing AZ AML approach of using a Voting Ensemble of algorithms proved to be .9177 accurate
    - To judge the AML model a little closer, see the following annotated images:
        - As mentioned, the best performing algorithm given by AZ AML in terms of accuracy was an Ensembling method.
        ![Ensemble Methods](./AML/AML_sorted.png)
        - In total, the ensemble was made up of 10 algorithms. The incorporated algorithms made use of 2 established scaling techniques as well as varrying estimation techniques ranging from GLM over Tree based algorithms to Boosted methods. A discussion of all of these would be out of the scope of this short documentation, however,the following image provides a quick overview: 
        ![Ensemble Details](./AML/AML_ensembling_details.png)
        - Regarding the metrics of the AML model, the confusion table sheds more light on the reliability of the accuracy measure:
        ![AML Confusion Table](./AML/AML_confusion_table.png)
            - While the high accuracy of over 90% suggest a very reliable classification estimation, the false positive/negative estimations demand closer attention. This is corroberated by the presence of a majority class, which can also be seen in the balanced accuracy. Significantly lower than the "unbalanced" accuracy, imbalanced training classes hence effectively render out of sample predictions less reliable.
            ![AML Accuracy](./AML/AML_balanced_acc.png)
            
## Future work
- Not only could you infer imbalance from the metrics provided by AZ AML, AZ AML also prompted a warning in relation to class imbalance when spinning up the AML command
- The presence of imbalance will effectively bias predictions made by the classification model towards the majority class
- As such, to improve classification for out of sample observations, SMOTE or some other method of handling class imbalance seems promising
- It might furthermore be worth to check out other metrics to ensure reliable out of sample classifications