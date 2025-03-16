# Stroke Prediction Using Machine Learning

This repository contains a concise summary of a project that predicts the likelihood of stroke in individuals based on their health and demographic data. The goal is to identify people at risk of stroke early on, thereby enabling timely interventions.

---

## 1. Overview
- **Objective**: Predict stroke risk (yes/no) from health and demographic variables.
- **Dataset**: Kaggle’s “Stroke Prediction Dataset” with 5110 records, containing features such as:
  - **Demographics**: Gender, Age, Residence Type, Work Type  
  - **Health Factors**: Hypertension, Heart Disease, BMI, Average Glucose Level  
  - **Lifestyle Factors**: Smoking Status  
- **Motivation**: Stroke is a leading cause of death worldwide. Accurate prediction enables preventative measures and improved healthcare outcomes.

---

## 2. Data Preprocessing
1. **Missing Values**: Mean imputation for missing BMI values (201 entries).  
2. **Categorical Encoding**: Converted categories (gender, smoking status, etc.) into numerical form.  
3. **Feature Selection**: Removed non-essential features like `id` and `ever_married`; kept age, BMI, glucose levels, hypertension, and heart disease, among others.  
4. **Imbalanced Classes**: Applied SMOTE (Synthetic Minority Oversampling Technique) to address the imbalance of positive (stroke) vs. negative (non-stroke) samples.  
5. **Scaling**: Applied standard scaling (`StandardScaler`) to continuous features (age, avg_glucose_level, bmi).

---

## 3. Methods

### Logistic Regression
- Well-established for binary classification (stroke vs. no stroke).
- Produces probability scores for each class.
- Hyperparameters included `C=0.1` and L2 regularization.

### Random Forest
- Ensemble approach using multiple decision trees.
- Tuned hyperparameters (`max_features=2`, `n_estimators=100`, `max_depth=5`).
- Often handles non-linear relationships better.

### Model Validation
- **Train/Test Split**: 80% for training, 20% for testing.
- **Cross-Validation**: 10-fold cross-validation to reduce risk of overfitting.
- **Performance Metrics**:
  - **Recall** (sensitivity): Critical in medical contexts to avoid false negatives.
  - **Log-Loss**: Evaluates how well predicted probabilities align with actual labels.

---

## 4. Results

- **Cross-Validation (mean recall)**:  
  - Random Forest: ~0.9072  
  - Logistic Regression: ~0.8294

- **Test Set**:
  - Recall  
    - Random Forest: 0.6429  
    - Logistic Regression: 0.6190  
  - Log-Loss  
    - Random Forest: 8.7816  
    - Logistic Regression: 9.2048  

Random Forest showed slightly higher recall and lower log-loss, making it the preferred model.

---

## 5. Conclusion
- The Random Forest model achieved better overall performance and was chosen as the final classifier.
- Future improvements could include:
  - Additional features (e.g., other biomarkers).
  - More extensive hyperparameter tuning.
  - Advanced methods for dealing with outliers and class imbalance.

---

## References
1. [Risk factors for stroke (Wang et al.)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6288566/)
2. [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
3. [SMOTE in Imbalanced-learn](https://imbalanced-learn.org/stable/references/api.html#imblearn.over_sampling.SMOTE)
4. [StandardScaler in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
5. [RandomForestClassifier in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
6. [DataFrame.sample in pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html)
7. [train_test_split in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

**Note**: This README is a condensed version of the original report, focusing on the essential steps and findings of the stroke prediction project.
