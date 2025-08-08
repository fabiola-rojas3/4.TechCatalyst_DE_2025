# **Lab Assignment: Customer Churn Prediction with PyCaret (AutoML)**

## **Case Study Introduction**

You have just joined the **Data Science team** at a telecommunications company.
Management is concerned about **customer churn** — the rate at which customers stop using the company’s services.
High churn means **loss of revenue** and **increased costs** to acquire new customers.

Your role as the **data scientist** is to:

1. Analyze historical customer data.
2. Build predictive models to **identify customers at risk of churning**.
3. Provide insights into the most important factors influencing churn.
4. Present results and recommendations to business stakeholders.

We will use the **Telco Customer Churn** dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## **Your Mission**

You will:

- **Download & load** the dataset into Python using Pandas.
- **Explore & understand** the dataset (EDA).
- **Split** the dataset into train and test sets.
- **Train** models using PyCaret’s AutoML capabilities.
- **Evaluate & compare** models.
- **Analyze** the top 3 models for:
  - Feature importances.
  - Performance on unseen test data.
  - Interpretation of results.
- **Understand** the preprocessing steps PyCaret performed automatically.
- **Document** the entire workflow in Markdown and publish it on GitHub.

---

## **Step-by-Step Instructions**

### **1. Get the Dataset**

1. Create a **Kaggle account** (if you don’t have one).
2. Download the dataset ZIP file from:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
3. Unzip it and save the CSV file (usually `WA_Fn-UseC_-Telco-Customer-Churn.csv`) into your project folder.

---

### **2. Load Data with Pandas**

```python
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
```

------

### **3. Exploratory Data Analysis (EDA)**

**EDA** (Exploratory Data Analysis) is the process of **examining your dataset** to understand its characteristics before modeling.

**Things to check:**

- **Shape** of the dataset (`df.shape`)
- **Column names & data types** (`df.info()`)
- **Missing values** (`df.isnull().sum()`)
- **Basic statistics** (`df.describe()`)
- **Class balance**: Is `Churn` evenly split between "Yes" and "No"?

```python
df['Churn'].value_counts(normalize=True)
```

If one class is much larger, it’s called **class imbalance** — and it will affect model performance.

------

### **4. Split Data into Train and Test Sets**

We will keep **1000 records** aside for final testing (hold-out set).

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=1000, random_state=42, stratify=df['Churn'])
```

- **Training set**: Used for model building.
- **Test set**: Used only **once** at the very end to check real-world performance.

------

### **5. Train with PyCaret**

**Question:** What type of ML problem is this?

- Hint: The target variable `Churn` is categorical ("Yes"/"No") → **Classification problem**.

```python
from pycaret.classification import *

# Initialize setup
clf_setup = setup(
    data=train_df, 
    target='Churn', 
    session_id=42,
    train_size=0.8
)

# Compare models
best_models = compare_models(n_select=3)
```

------

### **6. Analyze Results**

- Look at metrics like **Accuracy**, **Precision**, **Recall**, **F1-score**, **AUC**.
- **Confusion Matrix**:

```python
plot_model(best_models[0], plot='confusion_matrix')
```

------

### **7. Feature Importance**

```python
plot_model(best_models[0], plot='feature')
```

- Which features matter most? Why might they influence churn?
  - Did the feature importance work for these models? Why or why not?
  - What models would be better fit for features importance (tree based models for example)
    - Test a tree based model like Extra Trees Classifier or Ada Boost Classifier or Random Forest Classifier 
    - Did that work?
      - What were the top features 

------

### **8. Test on the Unseen Test Set**

```python
predictions = predict_model(best_models[0], data=test_df)
predictions.head()
```

Repeat for all top 3 models and compare.

------

### **9. Understand PyCaret’s Preprocessing**

When you run `setup()`, PyCaret:

- Detects data types.
- Handles missing values.
- Encodes categorical features.
- Scales/normalizes numeric features.
- Splits train/validation automatically.
- Creates a **pipeline**.

**Pipeline:**
A sequence of data transformation + model steps that ensures **exactly the same preprocessing** is applied to training and test data.

**Example pipeline flow:**

```mathematica
Raw Data
   ↓
Missing Value Imputation
   ↓
Categorical Encoding
   ↓
Scaling / Normalization
   ↓
Feature Selection / Transformation
   ↓
Model Training
```

------

### **10. Deliverable: One-Page GitHub Report**

Write a **Markdown (.md)** file summarizing your work:

**Sections:**

1. **Introduction**
   - Business problem.
   - Goal of the project.
2. **Data Understanding**
   - Dataset size, variables, missing values, class balance.
3. **Modeling**
   - ML problem type.
   - PyCaret setup.
   - Top 3 models chosen. 
     - What can you research and learn about these models? What are they used for? How to they compare to the others in the list.
4. **Evaluation**
   1. How did you decide the model is good?
      1. Confusion matrix.
      2. Metrics.
5. **Feature Importance**
   - Top features and their interpretation.
6. **PyCaret Preprocessing**
   - Steps performed automatically.
   - Pipeline diagram.
7. **Conclusion**
   - Which model would you recommend? Why?

Example pipeline diagram can be drawn with tools like draw.io or embedded as an image.

------

✅ **Final Check Before Submission:**

- Your `.ipynb` file runs without errors.
- Your `.md` file is clean and well-structured.
- You have uploaded both to your **GitHub repository**.