# 🛳️ Titanic Survival Prediction

## 📖 Project Overview
This project aims to predict whether a passenger aboard the Titanic survived or not, using demographic and socio-economic data.  
The analysis was carried out using **Python** and **machine learning models** implemented in **scikit-learn**.  
The project demonstrates a complete end-to-end machine learning workflow — from data preprocessing, through model tuning, to performance evaluation and feature interpretation.

---

## 📊 Dataset

The dataset used in this project is the **built-in Titanic dataset from the Seaborn library** (`sns.load_dataset('titanic')`).  
The image shown in the report only describes the dataset columns — no external data was used.

| Variable | Definition |
|-----------|-------------|
| survived | 0 = No, 1 = Yes |
| pclass | Ticket class (1, 2, 3) |
| sex | Gender of the passenger |
| age | Age in years |
| sibsp | # of siblings/spouses aboard |
| parch | # of parents/children aboard |
| fare | Ticket fare |
| embarked | Port of embarkation |
| class | Ticket class (string) |
| who | man, woman, or child |
| adult_male | True/False |
| alone | yes/no |

**Target variable:** `survived`  
**Feature variables:** `['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']`

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Missing numeric values were imputed using the **median**.
- Missing categorical values were imputed using the **most frequent** strategy.
- Numeric features were **standardized** using `StandardScaler`.
- Categorical features were **encoded** using `OneHotEncoder`.
- All transformations were combined into a **ColumnTransformer** within a pipeline.

### 2. Model Training
Two models were trained and compared:

#### 🔹 Random Forest Classifier
- Hyperparameters optimized using **GridSearchCV** with 5-fold **Stratified Cross-Validation**.
- Parameters tuned:  
  `n_estimators`, `max_depth`, `min_samples_split`.
- Achieved **test accuracy: ~82–85%**.

#### 🔹 Logistic Regression
- Tested with `liblinear` solver and both `L1` and `L2` penalties.
- Slightly lower accuracy than Random Forest, but much more interpretable.
- Used to analyze **feature coefficients** and their impact on survival.

---

## 📈 Results and Visualizations

### Confusion Matrix
The confusion matrix shows good model performance, with most passengers correctly classified:

| | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | 98 | 12 |
| Actual 1 | 18 | 51 |

This indicates a solid balance between **precision** and **recall** for both survival classes.

---

### Feature Importance (Random Forest)
The Random Forest model identified the most influential features in predicting survival:

1. **Fare** – higher ticket prices correlated with higher survival probability.  
2. **Age** – younger passengers were more likely to survive.  
3. **Sex (male/female)** – gender was a strong indicator; women had higher chances of survival.  
4. **Who (man/woman/child)** – demographic information helped refine predictions.  
5. **Pclass** – higher-class passengers (1st class) had better survival rates.  
6. **SibSp / Parch** – family relationships had moderate influence.  

These results align with historical accounts of the Titanic disaster — “**women and children first**.”

---

### Logistic Regression Coefficients
The logistic regression model provided interpretable coefficients that confirmed similar feature influences:
- Positive coefficients: increased the probability of survival (e.g., `sex_female`, `fare`).
- Negative coefficients: reduced the probability of survival (e.g., `sex_male`, `adult_male`, `pclass`).

---

## 🧠 Conclusions

- The **Random Forest model** achieved a strong test accuracy of approximately **82–85%**, confirming it can effectively classify survival outcomes.
- The **Logistic Regression model** achieved slightly lower accuracy but offered clearer interpretability of feature effects.
- Key predictors of survival:
  - **Gender (sex)** – females had higher survival probability.
  - **Ticket class (pclass/class)** – first-class passengers had better chances.
  - **Fare** – higher fares correlated with survival.
  - **Age** – younger passengers, especially children, were more likely to survive.
- The models reflect real historical patterns observed in the Titanic disaster, validating their predictive logic.

---

## 🧩 Technologies Used

- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib  

---
