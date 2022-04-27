# WineDataset EDA & ML 🍷

In [1]:
```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, HTML
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

/kaggle/input/wine-quality-dataset/WineQT.csv

<details>
  <summary>In [2]:</summary>
  
  ```
  !pip install pywaffle
  ```
  
  Collecting pywaffle
  Downloading pywaffle-0.6.4-py2.py3-none-any.whl (565 kB)
     |████████████████████████████████| 565 kB 4.4 MB/s            
  Requirement already satisfied: matplotlib in /opt/conda/lib/python3.7/site-packages (from pywaffle) (3.5.1)
  Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (4.28.4)
  Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (1.3.2)
  Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (21.3)
  Requirement already satisfied: pillow>=6.2.0 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (8.2.0)
  Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (3.0.6)
  Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (2.8.2)
  Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (0.11.0)
  Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pywaffle) (1.20.3)
  Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.7/site-packages (from python-dateutil>=2.7->matplotlib->pywaffle) (1.16.0)
  Installing collected packages: pywaffle
  Successfully installed pywaffle-0.6.4
  WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
  
</details>

## Importing required libraries 📚

In [3]:
```
import graphviz
import missingno as msno
from pywaffle import Waffle

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import sklearn.metrics as metrics
```

## Dataset Fields 🪟

- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **density**
- **pH**
- **sulphates**
- **alcohol**

Output variable (based on sensory data):
- **quality (score between 0 and 10)**

## Setting up Color Palette 🎨

In [4]:
```
colors = ['#00008B', '#1434A4', '#6082B6']
palette = sns.set_palette(sns.color_palette(colors))
sns.palplot(sns.color_palette(colors), size=1)
plt.tick_params(axis='both')
```

![image](https://user-images.githubusercontent.com/55765292/165466738-b35a85a5-593e-4ca4-a7b6-356353b8dff8.png)

## Checking up the dataset 🗸

### Reading the dataset

In [5]:
```
df = pd.read_csv('../input/wine-quality-dataset/WineQT.csv')
```

In [6]:
```
df
```

Out[6]:
![image](https://user-images.githubusercontent.com/55765292/165467125-68573418-789d-4f54-9134-1c641dc32675.png)

In [7]:
```
df.info()
```

![image](https://user-images.githubusercontent.com/55765292/165467196-6f2cf5ba-05ce-4adc-99ec-ca3a11314915.png)

### Checking null values

In [8]:
```
pd.DataFrame({'Missing Values': df.isna().sum()})
```

Out[8]:

![image](https://user-images.githubusercontent.com/55765292/165467281-c54e5d21-a37e-4e3b-a40e-9d8e5e682165.png)

### Statistical summary of the dataset

In [9]:
```
df.describe().rounbd(2)
```

Out[9]:

||fixed acidity|volatile acidity|citric acid|residual sugar|chlorides|free sulfur dioxide|total sulfur dioxide|density|pH|sulphates|alcohol|quality|Id|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|count|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|1143.00|
|mean|8.31|0.53|0.27|2.53|0.09|15.62|45.91|1.00|3.31|0.66|10.44|5.66|804.97|
|std|1.75|0.18|0.20|1.36|0.05|10.25|32.78|0.00|0.16|0.17|1.08|0.81|464.00|
|min|4.60|0.12|0.00|0.90|0.01|1.00|6.00|0.99|2.74|0.33|8.40|3.00|0.00|
|25%|7.10|0.39|0.09|1.90|0.07|7.00|21.00|1.00|3.20|0.55|9.50|5.00|411.00|
|50%|7.90|0.52|0.25|2.20|0.08|13.00|37.00|1.00|3.31|0.62|10.20|6.00|794.00|
|75%|9.10|0.64|0.42|2.60|0.09|21.00|61.00|1.00|3.40|0.73|11.10|6.00|1209.50|
|max|15.90|1.58|1.00|15.50|0.61|68.00|289.00|1.00|4.01|2.00|14.90|8.00|1597.00|

## Dataset EDA 📊

### Checking the Wine Quality distribution

In [10]:
```
quality = df['quality'].value_counts()

fig = plt.figure(
    FigureClass = Waffle, 
    rows = 4,
    columns = 8,
    values = quality,
    labels = ['{} - {}'.format(a, b) for a, b in zip(quality.index, quality)],
    legend = {
        'loc': 'upper left', 
        'bbox_to_anchor': (1, 1), 
        'fontsize': 20, 
        'labelcolor': 'linecolor',
        'title': 'Wine Quality',
        'title_fontsize': 20
        },
    font_size = 60, 
    icon_legend = True,
    figsize = (10, 8)
)

plt.title('Wine Quality Distribution', fontsize = 20)
plt.show()
```

![image](https://user-images.githubusercontent.com/55765292/165467858-a220f814-4635-4a9e-b2cb-581fd5673120.png)

In [11]:
```
def feature_viz(feature):
    plt.figure(figsize=(15,8))
    plt.title(f'{feature} hist plot')
    plt.subplot(1,3,1)
    df[feature].plot(kind='hist')

    plt.subplot(1,3,2)
    plt.title(f'{feature} box plot')
    sns.boxplot(df[feature])

    plt.subplot(1, 3, 3)
    plt.title(f'{feature} density plot')
    sns.kdeplot(df[feature])
    plt.tight_layout()
```

### Checking whether the distribution is normal and checking the number of outliers

In [12]:
```
for i in df.columns:
    feature_viz(i)
```
![image](https://user-images.githubusercontent.com/55765292/165470108-3dec8624-f3c7-4ed3-bcff-33c250204e89.png)

![image](https://user-images.githubusercontent.com/55765292/165470115-81e3bb02-f608-49fd-b5bf-6515ce72287d.png)

![image](https://user-images.githubusercontent.com/55765292/165470262-ffdaccc7-46b5-4461-b3ae-a94576b6d7da.png)

![image](https://user-images.githubusercontent.com/55765292/165470341-b43095a9-544c-42a8-a109-800e6b4043a3.png)

![image](https://user-images.githubusercontent.com/55765292/165470436-6f100a35-8897-48d1-8873-74b5b54273ce.png)

![image](https://user-images.githubusercontent.com/55765292/165470497-413cc5d2-437f-453e-8ce9-23b774aa47a0.png)

![image](https://user-images.githubusercontent.com/55765292/165470556-cdbf1eae-c0cb-4c15-8cfe-24ec63955119.png)

![image](https://user-images.githubusercontent.com/55765292/165470614-74ae80f4-11c9-430a-bc2f-bc18c1fcacd9.png)

![image](https://user-images.githubusercontent.com/55765292/165470657-b9d8a1a8-1396-45fd-ad79-742647fcee6d.png)

![image](https://user-images.githubusercontent.com/55765292/165470693-4533773a-8b94-4c56-8aa7-21181779de48.png)

![image](https://user-images.githubusercontent.com/55765292/165470733-f715a1b0-7bce-42a9-8b29-c879ed11dac6.png)

![image](https://user-images.githubusercontent.com/55765292/165470796-a2bd446b-d0d9-4578-b309-eac073c6d4d3.png)

![image](https://user-images.githubusercontent.com/55765292/165470824-3616e979-0790-48a0-b506-3efd68e8d4f1.png)

As there are extreme outliers in some features, I'll use RobustScaler and log transformation for normalizing them.


### Checking the correlation plot

In [17]:
```
mask = np.zeros_like(df.drop('Id', axis=1).corr())
tri_ind = np.triu_indices_from(mask)
mask[tri_ind] = True
plt.figure(figsize=[15, 10])
sns.heatmap(data=df.drop('Id', axis=1).corr(), annot=True, mask=mask, cmap='Blues', square=True)
```

Out[13]:
![image](https://user-images.githubusercontent.com/55765292/165470998-74affad8-734f-4af5-ba43-e01b93e946d7.png)

### Correlation with Target feature

In [14]:
```
plt.figure(figsize=(14,6))
corr = df.corr()['quality'].sort_values(ascending=False)
corr.drop('quality').plot(kind='bar')
```

Out[14]:
![image](https://user-images.githubusercontent.com/55765292/165471122-34386cf5-4efc-488b-8169-61342995a97a.png)

I'll drop 'quality','free sulfur dioxide', 'pH', 'residual sugar' as they have low correlation and take log transformation of remaining features.

In [15]:
```
x = np.log1p(df.drop(['quality','free sulfur dioxide', 'pH', 'residual sugar'], axis=1))
y = df['quality']
```

## Splitting the data 🪓

In [16]:
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
```

In [17]:
```
rs = RobustScaler()
x_train = rs.fit_transform(x_train)
x_test = rs.transform(x_test)
```

## Performing the training and testing session 🧪

### Logistic Regression

In [18]:
```
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_acc = lr.score(x_test, y_test)
print("Training accuracy for Logistic Regression: ", lr.score(x_train, y_train)*100, "%")
print("Testing accuracy for Logistic Regression:", lr_acc*100, "%")
```

Training accuracy for Logistic Regression:  58.75 %
Testing accuracy for Logistic Regression: 65.59766763848397 %

### K-Nearest Neighbors

In [19]:
```
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_acc = knn.score(x_test, y_test)
print("The training accuracy for KNN is:", knn.score(x_train, y_train)*100, "%")
print("The testing accuracy for KNN is:", knn_acc * 100, "%")
```

The training accuracy for KNN is: 69.25 %
The testing accuracy for KNN is: 52.76967930029155 %

### SVC

```
svc = SVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
svc_acc = svc.score(x_test, y_test)
print("The training accuracy for SVC is:", svc.score(x_train, y_train)*100, "%")
print("The testing accuracy for SVC is:", svc_acc * 100, "%")
```

The training accuracy for SVC is: 64.875 %
The testing accuracy for SVC is: 65.3061224489796 %

### Decision Tree Classifier

In [21]:
```
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_pred = dtc.predict(x_test)
dtc_acc = dtc.score(x_test, y_test)
print("The training accuracy for decision tree classifier is:", dtc.score(x_train, y_train)*100, "%")
print("The testing accuracy for decision tree classifier is:", dtc_acc * 100, "%")
```

The training accuracy for decision tree classifier is: 100.0 %
The testing accuracy for decision tree classifier is: 56.268221574344025 %

### Visualizing the Decision Tree graph

In [22]:
```
dot_data = tree.export_graphviz(dtc, out_file = None, feature_names = df.drop(['quality','free sulfur dioxide', 'pH', 'residual sugar'], axis=1).columns, class_names = ["3", "4", "5", "6", "7", "8"], filled = True)
graph = graphviz.Source(dot_data, format = "jpg")
display(graph)
```

![image](https://user-images.githubusercontent.com/55765292/165472657-b2f64489-2d88-4f5e-9a45-1c9108da8fb0.png)

### Random Forest

In [23]:
```
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
rfc_acc = rfc.score(x_test, y_test)
print("The training accuracy for Random Forest is:", rfc.score(x_train, y_train)*100, "%")
print("The testing accuracy for Random Forest is:", rfc_acc * 100, "%")
```

The training accuracy for Random Forest is: 100.0 %
The testing accuracy for Random Forest is: 69.67930029154519 %

### Adaboost

In [24]:
```
abc = AdaBoostClassifier()
abc.fit(x_train, y_train)
abc_pred = abc.predict(x_test)
abc_acc = abc.score(x_test, y_test)
print("The training accuracy for AdaBoost is:", abc.score(x_train, y_train)*100, "%")
print("The testing accuracy for AdaBoost is:", abc_acc * 100, "%")
```

The training accuracy for AdaBoost is: 53.25 %
The testing accuracy for AdaBoost is: 58.89212827988338 %

### Extra Trees Classifier

In [25]:
```
etc = ExtraTreesClassifier()
etc.fit(x_train, y_train)
etc_pred = etc.predict(x_test)
etc_acc = etc.score(x_test, y_test)
print("The training accuracy for Extra Trees Classifier is:", etc.score(x_train, y_train)*100, "%")
print("The testing accuracy for Extra Trees Classifier is:", etc_acc * 100, "%")
```

The training accuracy for Extra Trees Classifier is: 100.0 %
The testing accuracy for Extra Trees Classifier is: 69.09620991253644 %

### Bagging Classifier

In [26]:
```
bc = BaggingClassifier()
bc.fit(x_train, y_train)
bc_pred = bc.predict(x_test)
bc_acc = bc.score(x_test, y_test)
print("The training accuracy for bagging classifier is:", bc.score(x_train, y_train)*100, "%")
print("The testing accuracy for bagging classifier is:", bc_acc * 100, "%")
```

The training accuracy for bagging classifier is: 98.375 %
The testing accuracy for bagging classifier is: 66.1807580174927 %

### Gradient Boosting Classifier

In [27]:
```
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_pred = gbc.predict(x_test)
gbc_acc = gbc.score(x_test, y_test)
print("The training accuracy for gradient boosting classifier is:", gbc.score(x_train, y_train)*100, "%")
print("The testing accuracy for gradient boosting classifer is:", gbc_acc * 100, "%")
```

The training accuracy for gradient boosting classifier is: 94.625 %
The testing accuracy for gradient boosting classifer is: 61.51603498542274 %

### XGBoost Classifier

In [28]:
```
xgb = XGBClassifier(verbosity=0)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_acc = xgb.score(x_test, y_test)
print("The training accuracy for XGB is:", xgb.score(x_train, y_train)*100, "%")
print("The testing accuracy for XGB is:", xgb_acc * 100, "%")
```

The training accuracy for XGB is: 100.0 %
The testing accuracy for XGB is: 62.68221574344023 %

### Catboost Classifier

In [29]:
```
cbc = CatBoostClassifier(verbose=0)
cbc.fit(x_train, y_train)
cbc_pred = cbc.predict(x_test)
cbc_acc = cbc.score(x_test, y_test)
print("The training accuracy for CatBoost Classifer is:", cbc.score(x_train, y_train)*100, "%")
print("The testing accuracy for CatBoost Classifier is:", cbc_acc * 100, "%")
```

The training accuracy for CatBoost Classifer is: 100.0 %
The testing accuracy for CatBoost Classifier is: 69.09620991253644 %

## Models Summary 📝

<details>
  <summary>In [30]:</summary>
  
  ```
  models = {'Logistic': lr_acc, 'KNN': knn_acc, 'SVC': svc_acc, 'Decision Tree': dtc_acc, 'Random Forest': rfc_acc, 'AdaBoost': abc_acc, 'Extra Trees': etc_acc, 'Bagging': bc_acc, 'Gradient Boosting': gbc_acc, 'XGB': xgb_acc, 'Catboost': cbc_acc}
  models_df = pd.DataFrame(pd.Series(models))
  models_df.columns = ['Scores']
  models_df['Name'] = ['Logistic', 'KNN', 'SVC', 'Decision Tree', 'Random Forest',  'AdaBoost', 'Extra Trees', 'Bagging', 'Gradient Boosting', 'XGB', 'Catboost']
  models_df.set_index(pd.Index([1, 2, 3 , 4, 5 , 6, 7, 8, 9, 10, 11]))
  ```

  Out[30]:
  ![image](https://user-images.githubusercontent.com/55765292/165473287-108fbf82-ff81-457c-87b5-5dc119df0ef2.png)
  
</details>

In [31]:
```
plt.figure(figsize=[15, 10])
axis = sns.barplot(x = 'Name', y = 'Scores', data = models_df)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
```

![image](https://user-images.githubusercontent.com/55765292/165473579-3f968960-d576-4734-8be5-533b9af7d801.png)

## I hope you like this!✌️


