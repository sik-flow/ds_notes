I will show how to make a linear regression in Sklearn and Statsmodels.   First I will use sklearn to make a regression dataset. 


```python
from sklearn.datasets import make_regression
import pandas as pd
```


```python
X, y = make_regression(n_features = 2, noise=10, random_state=11)
```


```python
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Y'] = y
```


```python
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
      <th>X1</th>
      <th>X2</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.217348</td>
      <td>0.117820</td>
      <td>35.528041</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.529372</td>
      <td>1.561704</td>
      <td>53.670233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.886240</td>
      <td>-0.475733</td>
      <td>-93.494490</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.713560</td>
      <td>-1.908290</td>
      <td>-142.676470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.297423</td>
      <td>-0.714962</td>
      <td>-107.748177</td>
    </tr>
  </tbody>
</table>
</div>



### Sklearn


```python
from sklearn.linear_model import LinearRegression
import numpy as np

lr = LinearRegression()
lr.fit(df[['X1', 'X2']], df['Y'])

# Coefficients
lr.coef_
```




    array([60.05070199, 59.28817607])




```python
# Intercept 
lr.intercept_
```




    -0.4812452912200803




```python
# Prediction for X1 = 0.5 and X2 = 0.5
lr.predict(np.array([.5, .5]).reshape(1, -1))
```




    array([59.18819374])




```python
# R^2
lr.score(df[['X1', 'X2']], df['Y'])
```




    0.9846544787076148




```python
# Adjusted R^2
R2 = lr.score(df[['X1', 'X2']], df['Y'])
n = len(df)
p = 2

1-(1-R2)*(n-1)/(n-p-1)
```




    0.9843380762067409




```python
# mean squared error 
from sklearn.metrics import mean_squared_error
mean_squared_error(df['Y'], lr.predict(df[['X1', 'X2']]))
```




    95.90101789061725



### Statsmodels formula 


```python
from statsmodels.formula.api import ols
formula = 'Y ~ X1 + X2'
model = ols(formula=formula, data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.985</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.984</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3112.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>1.05e-88</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:12:37</td>     <th>  Log-Likelihood:    </th> <td> -370.06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   746.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    97</td>      <th>  BIC:               </th> <td>   753.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -0.4812</td> <td>    0.994</td> <td>   -0.484</td> <td> 0.630</td> <td>   -2.455</td> <td>    1.492</td>
</tr>
<tr>
  <th>X1</th>        <td>   60.0507</td> <td>    1.052</td> <td>   57.082</td> <td> 0.000</td> <td>   57.963</td> <td>   62.139</td>
</tr>
<tr>
  <th>X2</th>        <td>   59.2882</td> <td>    1.059</td> <td>   56.000</td> <td> 0.000</td> <td>   57.187</td> <td>   61.389</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.469</td> <th>  Durbin-Watson:     </th> <td>   1.874</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.791</td> <th>  Jarque-Bera (JB):  </th> <td>   0.619</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.128</td> <th>  Prob(JB):          </th> <td>   0.734</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.711</td> <th>  Cond. No.          </th> <td>    1.08</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The coefficients, intercept, R^2 and adjusted R^2 are all in the summary 


```python
# Prediction for X1 = 0.5 and X2 = 0.5
model.predict(dict(X1 = 0.5, X2 = 0.5))
```




    0    59.188194
    dtype: float64




```python
# mean squared error 
mean_squared_error(df['Y'], model.predict(dict(X1 = df['X1'].values, X2 = df['X2'].values)))
```




    95.90101789061725



### Statsmodels API


```python
import statsmodels.api as sm

X = df[['X1', 'X2']]
Y = df['Y']

# coefficient 
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.985</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.984</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3112.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>1.05e-88</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:16:34</td>     <th>  Log-Likelihood:    </th> <td> -370.06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   746.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    97</td>      <th>  BIC:               </th> <td>   753.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -0.4812</td> <td>    0.994</td> <td>   -0.484</td> <td> 0.630</td> <td>   -2.455</td> <td>    1.492</td>
</tr>
<tr>
  <th>X1</th>    <td>   60.0507</td> <td>    1.052</td> <td>   57.082</td> <td> 0.000</td> <td>   57.963</td> <td>   62.139</td>
</tr>
<tr>
  <th>X2</th>    <td>   59.2882</td> <td>    1.059</td> <td>   56.000</td> <td> 0.000</td> <td>   57.187</td> <td>   61.389</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.469</td> <th>  Durbin-Watson:     </th> <td>   1.874</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.791</td> <th>  Jarque-Bera (JB):  </th> <td>   0.619</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.128</td> <th>  Prob(JB):          </th> <td>   0.734</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.711</td> <th>  Cond. No.          </th> <td>    1.08</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The coefficients, intercept, R^2 and adjusted R^2 are all in the summary 


```python
# Prediction for X1 = 0.5 and X2 = 0.5
# Have to add a 1 in the front due to the y intercept 
Xnew = np.column_stack([1, .5, .5])
model.predict(Xnew)
```




    array([59.18819374])




```python
# mean squared error 
mean_squared_error(df['Y'], model.predict(X))
```




    95.90101789061725


