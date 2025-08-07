# Lab: Handling Missing Data in Time Series

**Goal**: You will learn how to identify and manage missing values in time series datasets. We will explore various techniques, from simple statistical methods to more advanced imputation and interpolation

## Objective

Work in groups to explore, identify, and handle missing data in time series datasets using Python. By the end of this lab, you will:

- Perform data quality checks to identify missing values.
- Understand the difference between `NaN` and `NaT`.
- Explore common strategies for handling missing data: imputation and interpolation.
- Apply these strategies to two real-world datasets.

## Datasets

1. **CO2 Emissions Dataset:** Annual CO2 emissions report ([source](https://ourworldindata.org/co2-emissions))
2. **e-Shop Clickstream Dataset:** Online shopping clickstream ([UCI ML repo](https://archive.ics.uci.edu/ml/datasets/Online+Retail))

Both datasets are included in your lab environment:

- `co2_missing_only.csv`
- `clicks_missing.csv`

The original datas without missing data 

* `co2_original.csv`
* `clicks_original.csv`

------

## Setup

Install required packages if missing:

```
pip install matplotlib numpy statsmodels scikit-learn scipy pandas
```

Import libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

Here are some helper functions. You will need to create a `helper.py` files, and import the functions from that file inside your notebook

```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('grayscale')
plt.rc("figure", figsize=(16, 5))

def read_dataset(folder, file, date_col=None, format=None, index=False):
    '''Reads a CSV dataset and converts date columns to datetime.'''
    index_col = date_col if index else None
    df = pd.read_csv(folder / file, 
                     index_col=index_col, 
                     parse_dates=[date_col],
                     date_format=format)
    return df

def plot_dfs(df1, df2, col, title=None, xlabel=None, ylabel=None):
    '''Creates comparative plots of original vs. missing data.
	    hint: df1 is the original dataset, while df2 contains your DataFrame
	    with different imputation strategies
    '''
    df_missing = df2.rename(columns={col: 'missing'})
    columns = df_missing.loc[:, 'missing':].columns.tolist()
    subplots_size = len(columns)
    
    fig, ax = plt.subplots(subplots_size + 1, 1, sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    if title:
        fig.suptitle(title)
    
    df1[col].plot(ax=ax[0], figsize=(12, 10))
    ax[0].set_title('Original Dataset')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)    
    
    for i, colname in enumerate(columns):
        df_missing[colname].plot(ax=ax[i + 1])
        ax[i + 1].set_title(colname)
    
    fig.tight_layout()
    plt.show()

def rmse_score(df1, df2, col=None):
    '''Calculates RMSE scores between original and processed data.'''
    if col is None:
        raise ValueError("Column name must be specified")
        
    df_missing = df2.rename(columns={col: 'missing'})
    columns = df_missing.loc[:, 'missing':].columns.tolist()
    
    if len(columns) <= 1:
        raise ValueError("No comparison columns found")
    
    scores = []
    for comp_col in columns[1:]:
        rmse = np.sqrt(np.mean((df1[col] - df_missing[comp_col])**2))
        scores.append(rmse)
        print(f'RMSE for {comp_col}: {rmse}')
    
    return scores
```

> [!Tip]
>
> What do you think these functions are doing. These were written by a previous employee, you are asked to use them, but there is no documentation. 
>
> * `read_dataset`
> * `plot_dfs`
> * `rmse_score`
>   * What is RMSE (Root Mean Square Error)

------

## Part 1: Data Quality Checks

### 1.1 Load the Data

In this section, we will load our datasets and perform initial quality checks to identify missing data. We will use two datasets: one on **CO2 emissions** and another on **e-commerce clickstreams**.

```python
co2_df = # using the read_dataset
ecom_df = #using the read_dataset
```

### 1.2 Inspect the Data

- Use `.info()`, `.head()`, and `.shape` to explore each dataframe.
- **Group Task:** What columns have missing values? How many missing values are there in each column?

```Python
print(co2_df.info())
print(co2_df.head())
print(ecom_df.info())
print(ecom_df.head())
```

------

## Part 2: Identifying Missing Data

### 2.1 Quantify Missingness

- Use `.isnull().sum()` and `.isna().sum()` on both datasets (including the index).
- Use `.isnull().values.any()` for a quick check.

**Group Task:**

- What’s the difference between `NaN` and `NaT`?
- Do missing values appear in both data columns and date columns?

------

## Part 3: Handling Missing Data

### 3.1 Imputation Strategies

Discuss and experiment with the following:

#### **Univariate Imputation:** Fill missing values with mean, median, or mode. Try each option and compare the results. 

- You will apply three common imputation methods:
  - **Forward Fill (ffill):** Propagates the last valid observation forward.
  - **Backward Fill (bfill):** Uses the next valid observation to fill the gap.
  - **Mean Imputation:** Fills missing values with the mean of the column.

```Python
# Example
co2_df['co2_filled_mean'] = # YOUR CODE 
ecom_df['price_filled_median'] = # YOUR CODE
```

For each dataset, you will use different imputation method. You will need to append the column (add a column) to the dataset, this way you can plot them and compare their performance against the original dataset. 

Here is an example of how you can use the `plot_dfs` and the `rmse_score` functions

```python
plot_dfs(co2_original, co2_missing, 'co2', 
         title="Annual CO2 Emission per Capita", 
         xlabel="Years", 
         ylabel="x100 million tons")

_ = rmse_score(co2_original, co2_missing, 'co2')
```

**Group Task**:

* In what scenarios would forward fill be more appropriate than mean imputation for time series data?
* What does the term **Univariate** even mean? What is the difference between **Univariate Imputation vs Multivariate Imputation**
* Research other simple imputation techniques like median or mode imputation. How might they compare in this context? 

#### **Univariate Interpolation:** Fill missing values using time-based or linear interpolation.

While imputation methods like `ffill` or `mean` are useful, **interpolation** can often provide a more nuanced way to fill missing values in a time series. Interpolation estimates missing values based on the values of surrounding data points.

We will use the `.interpolate()` method in pandas to apply several common techniques

```Python
co2_df['linear'] = # YOUR CODE 
```

When you use interpolation for each data set, first `co2_df` then the `clicks_missing` DataFrame, make sure to create a column to reflect the interpolation methods you selected.

**Visualizing & Comparing Approaches** for each dataset against the different methods.

**Group Task:**

- Which strategy works best for the different datasets? Why?
  - Which interpolation method performed the best (lowest RMSE)? Why might that be the case for this dataset?
  - Which inspection method gave you better intuition on what worked vs what didn’t work (think RMSE vs Plot)
- How do the filled values compare to the original?
- The `quadratic` and `cubic` methods are more complex. When does it make sense to use such methods?

------

## Part 4: Multivariate Imputation using Scikit-Learn:

* What does it mean to do **Multivariate Imputation**?

```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import (ElasticNet, 
                                  LinearRegression, 
                                  BayesianRidge)
from sklearn.neighbors import KNeighborsRegressor
```

```python
folder = Path('../../datasets/Ch7/')

clicks_original = read_dataset(folder, 
                            'clicks_original.csv', 'date', index=True)
clicks_missing = read_dataset(folder, 
                            'clicks_missing.csv', 'date', index=True)
```



At a high level what is:

* `LinearRegression`
* `ExtraTreesRegressor`
* `BaggingRegressor`
* `BayesianRidge`
* `KNeighborsRegressor`

In more simplistic way, I want to confirm their purpose and use-case. Would it be safe to consider them all as different **regression** algorithms? If so, what does **Regression** even mean?

```python
regressors = [
    ('bayesianRidge', BayesianRidge()),
    ('extra_trees', ExtraTreesRegressor(n_estimators=50, random_state=42)),
    ('bagging', BaggingRegressor(n_estimators=50, random_state=42)),
    ('elastic_net', ElasticNet()),
    ('linear_regression', LinearRegression()),
    ('knn', KNeighborsRegressor(n_neighbors=3))
]
```

```python
# Select the relevant columns ('price', 'location', 'clicks')
# These are the first three columns and 'clicks' is the third (index 2)
clicks_vals = clicks_missing.iloc[:,0:3].to_numpy()

for reg_name, reg in regressors:
    est = IterativeImputer(
                random_state=15,
                estimator=reg).fit(clicks_vals)
    clicks_missing[reg_name] = est.transform(clicks_vals)[: , 2]
```

```python
_ = rmse_score(clicks_original, clicks_missing, 'clicks')
```

Looking at the output, which model seems to be doing better?

```python
plot_dfs(clicks_original, clicks_missing, 'clicks')
```

Visually, which model seems to be doing a better job?

## Part 5: Discussion & Reflection

**Group Questions:**

- What are the risks of ignoring missing data?
- When would you prefer interpolation to imputation (and vice versa)?
- How would missing data impact a predictive model?

## Deliverables 

* **Each group shares key findings and answers with the class.** 

* **Each individual will need to publish a short report on their findings in Markdown published on their GitHub repo**. This includes reflection on their understanding of different options available to use for handling missing data. Whether it is was part of this lab, or additional based on their research 