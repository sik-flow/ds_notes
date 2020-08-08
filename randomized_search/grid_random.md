
I'm going to compare the performance of tuning the hyperparameters with [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).  First, I am going to create a dataset that has 10,000 samples and 20 features. 


```python
from sklearn.datasets import make_classification
import numpy as np 
import pandas as pd

# make dataset 
X, y = make_classification(n_samples = 10000, 
                           n_features=20, 
                           n_informative=4, 
                           n_redundant=0, 
                           random_state=11)

df = pd.DataFrame(X)
df['target'] = y
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.520216</td>
      <td>-0.299523</td>
      <td>1.697775</td>
      <td>0.152835</td>
      <td>-0.071976</td>
      <td>0.002353</td>
      <td>0.057001</td>
      <td>1.656589</td>
      <td>0.059377</td>
      <td>0.634026</td>
      <td>...</td>
      <td>0.230848</td>
      <td>-2.133668</td>
      <td>-0.658056</td>
      <td>0.227366</td>
      <td>-1.005542</td>
      <td>-0.533868</td>
      <td>-0.656252</td>
      <td>-1.167656</td>
      <td>-0.902226</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.252605</td>
      <td>1.432791</td>
      <td>1.561181</td>
      <td>-1.456888</td>
      <td>-0.325153</td>
      <td>-1.757407</td>
      <td>1.183243</td>
      <td>0.931166</td>
      <td>0.967256</td>
      <td>-1.833468</td>
      <td>...</td>
      <td>-1.644497</td>
      <td>1.259892</td>
      <td>1.355751</td>
      <td>-1.085283</td>
      <td>-1.347220</td>
      <td>-0.073796</td>
      <td>0.718362</td>
      <td>-2.334630</td>
      <td>1.531651</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-1.118205</td>
      <td>-0.335938</td>
      <td>-0.979303</td>
      <td>0.188338</td>
      <td>-0.346252</td>
      <td>-1.263341</td>
      <td>-1.037886</td>
      <td>-0.870959</td>
      <td>2.105311</td>
      <td>0.892956</td>
      <td>...</td>
      <td>0.794894</td>
      <td>0.796176</td>
      <td>0.193527</td>
      <td>-2.070266</td>
      <td>-1.183444</td>
      <td>-0.231885</td>
      <td>1.581976</td>
      <td>1.110054</td>
      <td>1.610723</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.334311</td>
      <td>1.568198</td>
      <td>-0.423843</td>
      <td>-0.962124</td>
      <td>1.060851</td>
      <td>-3.596107</td>
      <td>-0.416077</td>
      <td>-0.602925</td>
      <td>-0.523378</td>
      <td>0.834385</td>
      <td>...</td>
      <td>-0.636568</td>
      <td>-2.537476</td>
      <td>-0.355572</td>
      <td>1.032740</td>
      <td>0.195867</td>
      <td>-0.227352</td>
      <td>-0.332308</td>
      <td>0.813405</td>
      <td>-1.037039</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-0.803574</td>
      <td>-0.573973</td>
      <td>2.605967</td>
      <td>0.600801</td>
      <td>0.823409</td>
      <td>0.494084</td>
      <td>-0.398244</td>
      <td>1.332191</td>
      <td>0.273173</td>
      <td>1.089310</td>
      <td>...</td>
      <td>-1.030162</td>
      <td>-1.252967</td>
      <td>1.109795</td>
      <td>-1.197247</td>
      <td>-0.681647</td>
      <td>-0.786710</td>
      <td>0.833898</td>
      <td>-0.258752</td>
      <td>0.161887</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Next, I'm going to build a parameter grid for an random forest classifier.  I got the parameter grid from this great [article](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74). 


```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
```


```python
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Create the random grid
param_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
                'n_estimators': n_estimators}
```

Now I will find the best hyperparameters for the random forest using GridSearchCV. 


```python
clf = RandomForestClassifier()
```


```python
rf_gridsearch = GridSearchCV(estimator = clf, param_grid = param_grid,
                             cv = 3, verbose=2, n_jobs = -1)
```


```python
%%time
rf_gridsearch.fit(df.drop('target', axis = 1), df['target'])
```

    Fitting 3 folds for each of 720 candidates, totalling 2160 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:  8.8min
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed: 23.0min
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed: 43.3min
    [Parallel(n_jobs=-1)]: Done 997 tasks      | elapsed: 69.0min
    [Parallel(n_jobs=-1)]: Done 1442 tasks      | elapsed: 102.5min
    [Parallel(n_jobs=-1)]: Done 1969 tasks      | elapsed: 139.3min
    [Parallel(n_jobs=-1)]: Done 2160 out of 2160 | elapsed: 152.9min finished


    CPU times: user 14.3 s, sys: 627 ms, total: 14.9 s
    Wall time: 2h 32min 59s





    GridSearchCV(cv=3, error_score='raise-deprecating',
                 estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators='warn', n_jobs=None,
                                                  oob_score=False,
                                                  random_state=None, verbose=0,
                                                  warm_start=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                           110, None],
                             'max_features': ['auto', 'sqrt'],
                             'min_samples_split': [2, 5, 10],
                             'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400,
                                              1600, 1800, 2000]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=2)




```python
rf_gridsearch.best_score_
```




    0.9072



Using gridsearch took 2 hours and 32 minutes and got a accuracy of 90.7%.  Now I will try using RandomizedSearch. 


```python
rf_random = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = 100,
                               cv = 3, verbose=2, random_state=11, n_jobs = -1)
```


```python
%%time
rf_random.fit(df.drop('target', axis = 1), df['target'])
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  2.1min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed: 10.0min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 20.9min finished


    CPU times: user 7.08 s, sys: 190 ms, total: 7.27 s
    Wall time: 21min 2s





    RandomizedSearchCV(cv=3, error_score='raise-deprecating',
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators='warn',
                                                        n_jobs=None,
                                                        oob_sc...
                                                        verbose=0,
                                                        warm_start=False),
                       iid='warn', n_iter=100, n_jobs=-1,
                       param_distributions={'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [200, 400, 600, 800,
                                                             1000, 1200, 1400, 1600,
                                                             1800, 2000]},
                       pre_dispatch='2*n_jobs', random_state=11, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
rf_random.best_score_
```




    0.907



Using randomizedsearch I got an accuracy of 90.7% and it took 21 minutes to train. So, I got the same accuracy in significantly less training time using randomizedsearch. 
