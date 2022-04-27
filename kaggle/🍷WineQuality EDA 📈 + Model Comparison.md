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


## 3 | Exploratory Analysis

<details>
  <summary>In [7]:</summary>
  
  ```
  fig = go.Figure()

  for x in range(6):
      fig.add_trace(go.Box(
          x=df[df['quality']==x]['volatile acidity'],
          y=df[df['quality']==x]['quality'], name='Quality '+str(x)
      ))

  fig.update_layout(
      yaxis_title='quality', xaxis_title='volatile acidity'
  )
  fig.update_traces(orientation='h')
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165456672-4abca444-fdb4-4c84-9705-f198693a3bba.png)

**Insights**
- ```volatile acidity``` has a correlation value of 0.407
- It can be seen in the boxplot that lower values of ```volatile acidity``` has higher values of ```quality```


<details>
  <summary>In [8]:</summary>
  
  ```
  fig = px.scatter(df, x="total sulfur dioxide", y="free sulfur dioxide", color=df['quality'],  color_continuous_scale='Blues')
  fig.update_layout(legend_title_text='Quality')
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165457009-df2c225c-9d7d-4cdf-b8ab-4f9fe8061fab.png)

**Insights**
- Looks like higher levels of ```total sulfur dioxide``` means higher values of ```free sulfur dioxide```
- Low levels of ```free sulfur dioxide``` and ```total sulfur dioxide``` usually mean a better quality

<details>
  <summary>In [9]:</summary>
  
  ```
  fig = go.Figure()

  for x in range(6):
      fig.add_trace(go.Box(
          x=df[df['quality']==x]['citric acid'],
          y=df[df['quality']==x]['quality'], name='Quality '+str(x)
      ))

  fig.update_layout(
      yaxis_title='quality', xaxis_title='citric acid'
  )
  fig.update_traces(orientation='h')
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165457224-9c1aa716-e6d6-4951-9142-d9e6d977732a.png)

**Insights**
- ```critric acid``` has a correlation value of 0.241
- Wines with high levels of ```citric acid``` usually fall into the quality category of 0 and 5


<details>
  <summary>In [10]:</summary>
  
  ```
  fig = px.scatter(df, x="fixed acidity", y="density", color=df['quality'],  color_continuous_scale='Blues')
  fig.update_layout(legend_title_text='Quality')
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165457384-edc2b845-0930-44a0-9906-f8bbeed8c05a.png)

**Insights**
- Looks like higher levels of ```fixed acidity``` means higher values of ```density```
- No correlation with quality can be seen


<details>
  <summary>In [11]:</summary>
  
  ```
  fig = go.Figure()

  for x in range(6):
      fig.add_trace(go.Box(
          x=df[df['quality']==x]['sulphates'],
          y=df[df['quality']==x]['quality'], name='Quality '+str(x)
      ))

  fig.update_layout(
      yaxis_title='quality', xaxis_title='sulphates'
  )
  fig.update_traces(orientation='h')
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165457611-92c460bc-7500-443a-9e00-cd75b333236f.png)

**Insights**
- ```sulphates``` has a correlation value of 0.258
- It can be seen in the boxplot that wines of higher levels of sulphates usually have better quality

<details>
  <summary>In [12]:</summary>
  
  ```
  fig = px.scatter(df, x="citric acid", y="volatile acidity", color=df['quality'],  color_continuous_scale='Blues')
  fig.update_layout(legend_title_text='Quality')
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165457770-ce6c552c-98a0-407c-b1c8-c98e0bcd2d0d.png)

**Insights**
- Looks like ```citric acid``` and ```volatile acidity``` have an inverse relationship, i.e. high values of ```citric acid``` means low values of ```volatile acidity```
- Low levels of ```volatile acidity``` usually means a better quality, as seen in the ```volatile acidity```-```quality``` Boxplot before

<details>
  <summary>In [13]:</summary>
  
  ```
  fig = go.Figure()

  for x in range(6):
      fig.add_trace(go.Box(
          x=df[df['quality']==x]['alcohol'],
          y=df[df['quality']==x]['quality'], name='Quality '+str(x)
      ))

  fig.update_layout(
      yaxis_title='quality', xaxis_title='alcohol'
  )
  fig.update_traces(orientation='h')
  fig.show()
  ```

</details>

![image](https://user-images.githubusercontent.com/55765292/165458033-99b3ac5d-0ce5-4f5b-bbf6-45f98ce37018.png)

**Insight**
- ```alcohol``` has the greatest value of correlation, a correlation value of 0.485
- It can be seen in the boxplot that wines of higher levels of alcohol usually have better quality ratings

## 4 | Data Preprocessing

### 4.1 | Normalizing continuous features

<details>
  <summary>In [14]:</summary>
  
  ```
  df.describe().T[['min', 'max']][:-1].style.background_gradient(cmap='Blues')
  ```
  
</details>

Out[14]:

![image](https://user-images.githubusercontent.com/55765292/165458246-2891594f-83bc-4071-bdc1-0f6dc48afba6.png)

All features are continuous, but they all have different ranges, so I am normalizing them to be between 0 and 1

```
for col in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
df[col] = df[col]/df[col].max()
```

### 4.2 | Preparing Training and Validation arrays

Here I am creating arrays for features and labels

```
features = np.array(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])
labels = np.array(df['quality'])
```

Splitting the dataset- 20% for validation, and the rest 80% for training

```
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=0)
```

## 5 | Models

In [18]:
```
model_comparison = {}
```

### 5.1 | SVC

In [19]:
```
parameters = {'C': [6,8,10,12,14,16], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

svc_model = SVC()

clf = GridSearchCV(svc_model, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)
model_comparison['SVC'] = [accuracy_score(y_val,y_pred), f1_score(y_val,y_pred, average='weighted')]
print('\n')
print(classification_report(y_val,y_pred, zero_division=1))
```

Searching for best hyperparameters ...
Best Hyperparameters: {'C': 12, 'kernel': 'poly'}

![image](https://user-images.githubusercontent.com/55765292/165459024-33b30756-3c57-482e-99ce-32d951064280.png)

### 5.2 | DecisionTreeClassifier

In [20]:
```
parameters = {'max_depth': [5,10,15,20]}

Tree_model = DecisionTreeClassifier()

clf = GridSearchCV(Tree_model, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)
model_comparison['DecisionTreeClassifier'] = [accuracy_score(y_val,y_pred), f1_score(y_val,y_pred, average='weighted')]
print('\n')
print(classification_report(y_val,y_pred, zero_division=1))
```

Searching for best hyperparameters ...
Best Hyperparameters: {'max_depth': 5}

![image](https://user-images.githubusercontent.com/55765292/165459203-5ee61428-1be2-4959-bfe4-b6e5825a8dd2.png)

### 5.3 | KNeighborsClassifier

In [21]:
```
parameters = {'n_neighbors': [10,20,30,40,50]}

K_model = KNeighborsClassifier()

clf = GridSearchCV(K_model, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)
model_comparison['KNeighborsClassifier'] = [accuracy_score(y_val,y_pred), f1_score(y_val,y_pred, average='weighted')]
print('\n')
print(classification_report(y_val,y_pred, zero_division=1))
```

Searching for best hyperparameters ...
Best Hyperparameters: {'n_neighbors': 20}

![image](https://user-images.githubusercontent.com/55765292/165459282-e62334d4-da3f-4bc1-9d83-d69ebcb4730d.png)

### 5.4 | RandomForestClassifier

In [22]:
```
parameters = {'n_estimators': [160,180,200], 'max_depth':[18,20,22,24]}

rf = RandomForestClassifier()

clf = GridSearchCV(rf, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)
model_comparison['RandomForestClassifier'] = [accuracy_score(y_val,y_pred), f1_score(y_val,y_pred, average='weighted')]
print('\n')
print(classification_report(y_val,y_pred, zero_division=1))
```

Searching for best hyperparameters ...
Best Hyperparameters: {'max_depth': 20, 'n_estimators': 160}

![image](https://user-images.githubusercontent.com/55765292/165459345-6fda32bd-691b-4866-8928-4a8a4f527ab9.png)

### 5.5 | XGBoost

In [23]:
```
parameters = {'n_estimators': [100, 150, 200], 'max_depth':[16, 18, 20]}

xgboost = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

clf = GridSearchCV(xgboost, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)
model_comparison['XGBoost'] = [accuracy_score(y_val, y_pred), f1_score(y_val,y_pred, average='weighted')]
print('\n')
print(classification_report(y_val,y_pred, zero_division=1))
```

Searching for best hyperparameters ...
Best Hyperparameters: {'max_depth': 18, 'n_estimators': 100}

![image](https://user-images.githubusercontent.com/55765292/165459435-b14164fe-6ce1-4c1d-93de-6f3c9428a75a.png)

### 5.6 | Model Comparison

<details>
  <summary>In [24]:</summary>
  
  ```
  model_comparison_df = pd.DataFrame.from_dict(model_comparison).T
  model_comparison_df.columns = ['Accuracy', 'F1 Score']
  model_comparison_df = model_comparison_df.sort_values('F1 Score', ascending=True)
  model_comparison_df.style.background_gradient(cmap='Blues')
  ```
  
</details>

Out[24]:
![image](https://user-images.githubusercontent.com/55765292/165459569-1ddc15cf-e8f6-4884-ac9b-32a5ee47ff05.png)

<details>
  <summary>In [25]:</summary>
  
  ```
  fig = go.Figure(data=[
    go.Bar(name='F1 Score', y=model_comparison_df.index, x=model_comparison_df['F1 Score'], orientation='h'),
    go.Bar(name='Accuracy', y=model_comparison_df.index, x=model_comparison_df['Accuracy'], orientation='h')
  ])
  fig.update_layout(barmode='group')
  fig.show()
  ```
  
</details>

![image](https://user-images.githubusercontent.com/55765292/165459679-c5e87e19-cb9f-4085-aaed-a0c4645b3e9e.png)



![tumblr_o17frv0cdu1u9u459o1_500](https://user-images.githubusercontent.com/55765292/165459909-437d03bf-5542-4455-9fa4-56a3d93f2e37.gif)
