# 🍷WineQuality EDA 📈 + Model Comparison

## 1 | Importing libraries
- **For ML Models**: sklearn
- **For Data Processing**: numpy, pandas, sklearn
- **For Data Visualization**: matplotlib, seaborn, plotly


<details>
  <summary>In [1]:</summary>
  
  ```
  # For ML models
  from sklearn.linear_model import LinearRegression ,LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB
  from sklearn.ensemble import AdaBoostRegressor
  from sklearn.ensemble import RandomForestClassifier
  import xgboost as xgb
  from sklearn.svm import SVC ,SVR
  from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
  from sklearn.model_selection import GridSearchCV

  # For Data Processing
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split 

  # For Data Visualization
  import matplotlib.pyplot as plt
  import seaborn as sns
  import plotly.express as px
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  # Miscellaneous
  import os
  import random
  ```
</details>

## 2 | About the Dataset

### 2.1 | Reading the data

Here I am subtracting 3 from the quality column to change the range of quality column from 3-8 to 0-5

<details>
  <summary>In [2]:</summary>
  
  ```
  df = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv')
  del df['Id']
  df['quality'] = df['quality']-3
  df
  ```

</details>

Out[2]:

![image](https://user-images.githubusercontent.com/55765292/165423477-0fdfb38b-4f28-4cd7-9dca-efd8ea6b730c.png)

### 2.2 | Column Statistics

<details>
  <summary>In [3]:</summary>
  
  ```
  df.describe()[1:].T.style.background_gradient(cmap='Blues')
  ```
  
</details>

Out[3]:

![image](https://user-images.githubusercontent.com/55765292/165423548-d768a26e-c1a0-42e9-bae8-79dc575d1500.png)

### 2.3 | Distribution of Quality

<details>
  <summary>In [4]:</summary>
  
  ```
  fig = go.Figure(data=[go.Pie(labels=df['quality'].value_counts().index, values=df['quality'].value_counts(), hole=.3)])
  fig.update_layout(legend_title_text='Quality')
  fig.show()
  ```

</details>

![image](https://user-images.githubusercontent.com/55765292/165423676-725dd612-48dd-4a66-96df-70da2f4eae3f.png)

### 2.4 | Correlation Matrix

<details>
  <summary>In [5]:</summary>
  
  ```
  fig = px.imshow(df.corr(),color_continuous_scale="Blues")
  fig.update_layout(height=750)
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165423770-4f7d3546-9bf8-4731-9299-2cafbe23f10e.png)

### 2.5 | Distribution of Correlation of features

<details>
  <summary>In [6]:</summary>
  
  ```
  df_corr_bar = abs(df.corr()['quality']).sort_values()[:-1]
  fig = px.bar(df_corr_bar, orientation='h', color_discrete_sequence =['#4285f4']*len(df_corr_bar))
  fig.update_layout(showlegend=False)
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165423944-f2b3338c-6765-4167-974c-1ed7a5fdef8e.png)
