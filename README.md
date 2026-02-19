
**CDC Diabetes Health Indicators – Multiclass Classification**

**Overview**

This project applies machine learning (ML) and deep learning (DL) methods to the CDC Diabetes Health Indicators Dataset to classify individuals into:
	•	0 = Healthy
	•	1 = Pre-diabetic
	•	2 = Diabetic

The primary objective is to understand the relationship between lifestyle, demographic factors, and diabetes status while addressing the challenges of class imbalance in multiclass medical prediction problems.

The work evaluates multiple modelling strategies, including neural networks (NN), feature selection, hyperparameter tuning, and classical ML baselines.

⸻

**Dataset**

The dataset contains 35 features derived from healthcare statistics and lifestyle surveys of the U.S. population.

Feature Categories
	•	Demographic variables
	•	Self-reported lifestyle indicators
	•	Health status and access-to-care variables

**Target Variable**

Label	Class
0	Healthy
1	Pre-diabetic
2	Diabetic

The dataset is highly imbalanced, with pre-diabetics forming the smallest class.

⸻

**Methods**

Core Challenges
	•	Severe class imbalance
	•	Misleading accuracy in multiclass medical classification
	•	Low representation of pre-diabetics (clinically important class)

⸻

Solutions Applied
	•	Class weighting
	•	Batch normalization
	•	Dropout regularization
	•	Early stopping
	•	LASSO feature selection
	•	Hyperparameter tuning:
	•	Keras Tuner
	•	RandomizedSearchCV
	•	BayesSearchCV

⸻

Models Evaluated
	1.	Baseline Sequential Neural Network (Keras)
	2.	Neural Network with LASSO feature selection
	3.	Hyperparameter-tuned Neural Network
	4.	XGBoost Classifier (no LASSO)
	5.	XGBoost Classifier (with LASSO)
	6.	One-vs-Rest Logistic Regression (LASSO)
	7.	One-vs-One Logistic Regression

⸻

**Evaluation Metrics**

Due to class imbalance, performance was assessed using:
	•	Recall (Sensitivity)
	•	Precision
	•	F1-score (Macro and Micro)
	•	ROC-AUC (micro-averaged)

Key Focus
	•	Recall and F1-score for the pre-diabetic class
	•	Macro-averaged F1, which treats all classes equally

**Accuracy alone was considered insufficient and potentially misleading.**

⸻

**Results Summary**

Model	Accuracy	Recall	Precision	F1 Score	Notes
Baseline NN	0.64	0.85	0.81	0.81	Strong recall for minority class
NN + LASSO	0.65	—	—	—	No meaningful improvement
Tuned NN	0.68	—	—	—	ROC-AUC 0.84
XGBoost (tuned)	0.84	0.84	0.80	0.81	Best overall model
XGBoost + LASSO	0.68	0.68	0.83	0.73	Reduced performance
OvR Logistic (LASSO)	0.67	—	—	—	Competitive baseline
OvO Logistic	0.64	—	—	—	Lower performance


⸻

**Key Findings**
	•	Hyperparameter-tuned XGBoost achieved the strongest overall performance.
	•	Neural networks with class weighting performed competitively, especially for recall-critical medical use cases.
	•	LASSO feature selection alone did not improve neural network performance.
	•	Macro-averaged metrics are essential in imbalanced multiclass medical problems.
	•	Pre-diabetic classification remains intrinsically difficult, even with optimal setups.

⸻

**Technical Notes**
	•	Python 3.10 was used due to instability observed in Python 3.14.
	•	When using Keras Tuner, the correct tuned model must be explicitly loaded from the saved directory.

⸻

**Technologies Used**
	•	Python 3.10
	•	TensorFlow / Keras
	•	Keras Tuner
	•	XGBoost
	•	scikit-learn
	•	NumPy
	•	Pandas
	•	Seaborn

⸻

**Data Source**

CDC Diabetes Health Indicators Dataset

⸻

**Conclusion**

Class-weighted Neural Networks and tuned XGBoost classifiers are effective approaches for multiclass diabetes classification in imbalanced datasets.

For medical applications, recall and macro-F1 should be prioritised over raw accuracy to avoid missing clinically important cases.

