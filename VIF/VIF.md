Multicollinearity is where 3 or more features are highly correlated.  I am going to demonstrate how to calculate when you are experiencing multicollinearity using variance inflation factor (VIF).

To start I am going to use sklearn to make a dataset. 


```python
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
np.random.seed(11)
```


```python
X, y = make_regression(n_features = 10, noise=10, random_state=11, )
df = pd.DataFrame(X, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'])
df['Y'] = y
```

I am going to create a new feature that is a combination of the other features. 


```python
df['X11'] = df['X1'] - df['X2'] + df['X3'] + np.random.normal(scale = 15)
```

Now lets look at the correlation heat map to see if there is any highly correlated features. 


```python
import seaborn as sns 
sns.heatmap(df.corr(), vmin = -1, vmax = 1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c2ebcc590>




![png](VIF_files/VIF_6_1.png)



```python
sns.heatmap(np.abs(df.corr()) > 0.7)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c2e9c38d0>




![png](VIF_files/VIF_7_1.png)


We see that none of the features have a correlation above 0.70 or below -0.70.  Next, I am going to use VIF to determine if I am experiencing multicollinearity. 


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

Note: per this [open issue](https://github.com/statsmodels/statsmodels/issues/2376) on variance inflation factor you need to add a constant variable to your dataframe.   


```python
df = sm.add_constant(df)
```

I am going to test out all the features to see what the VIF value is.  I am skipping the first feature (the intercept term). 


```python
for i in range(1, 12):
    print(f'{df.drop("Y", axis = 1).columns[i]} {variance_inflation_factor(df.drop("Y", axis = 1).values, i)}')
```

    X1 inf
    X2 inf
    X3 inf
    X4 1.0410536144535685
    X5 1.0917593842615227
    X6 1.0955635908289258
    X7 1.0425296201253647
    X8 1.1118123228822059
    X9 1.051488347630868
    X10 1.1173223003485167
    X11 inf


I see I have multiple features that have a VIF value of `inf`.  Lets see what is causing that. 

Below is the formula for how VIF is calculated. 

$$ VIF = \frac{1}{1 - R^2} $$ 

From above, I got a VIF of `inf` for `X1`.  To calculate VIF for `X1` - I am going to build a regression using all the independent variables except for `X1` and use those features to predict `X1`. 


```python
X = df.drop(['X1', 'Y'], axis = 1)
Y = df['X1']

model = sm.OLS(Y, X).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>X1</td>        <th>  R-squared:         </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.305e+27</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Jul 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:55:53</td>     <th>  Log-Likelihood:    </th> <td>  2874.8</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>  -5728.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    89</td>      <th>  BIC:               </th> <td>  -5699.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>    
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
  <th>const</th> <td>  -26.2418</td> <td> 2.42e-13</td> <td>-1.08e+14</td> <td> 0.000</td> <td>  -26.242</td> <td>  -26.242</td>
</tr>
<tr>
  <th>X2</th>    <td>    1.0000</td> <td> 1.26e-14</td> <td> 7.92e+13</td> <td> 0.000</td> <td>    1.000</td> <td>    1.000</td>
</tr>
<tr>
  <th>X3</th>    <td>   -1.0000</td> <td> 1.18e-14</td> <td>-8.47e+13</td> <td> 0.000</td> <td>   -1.000</td> <td>   -1.000</td>
</tr>
<tr>
  <th>X4</th>    <td>-3.816e-16</td> <td> 8.28e-15</td> <td>   -0.046</td> <td> 0.963</td> <td>-1.68e-14</td> <td> 1.61e-14</td>
</tr>
<tr>
  <th>X5</th>    <td> 3.088e-16</td> <td> 9.08e-15</td> <td>    0.034</td> <td> 0.973</td> <td>-1.77e-14</td> <td> 1.84e-14</td>
</tr>
<tr>
  <th>X6</th>    <td> 2.984e-16</td> <td> 9.72e-15</td> <td>    0.031</td> <td> 0.976</td> <td> -1.9e-14</td> <td> 1.96e-14</td>
</tr>
<tr>
  <th>X7</th>    <td>-1.388e-17</td> <td> 8.44e-15</td> <td>   -0.002</td> <td> 0.999</td> <td>-1.68e-14</td> <td> 1.68e-14</td>
</tr>
<tr>
  <th>X8</th>    <td> 1.596e-16</td> <td> 8.42e-15</td> <td>    0.019</td> <td> 0.985</td> <td>-1.66e-14</td> <td> 1.69e-14</td>
</tr>
<tr>
  <th>X9</th>    <td>-6.635e-17</td> <td> 8.85e-15</td> <td>   -0.007</td> <td> 0.994</td> <td>-1.77e-14</td> <td> 1.75e-14</td>
</tr>
<tr>
  <th>X10</th>   <td>-2.533e-16</td> <td> 8.44e-15</td> <td>   -0.030</td> <td> 0.976</td> <td> -1.7e-14</td> <td> 1.65e-14</td>
</tr>
<tr>
  <th>X11</th>   <td>    1.0000</td> <td>  9.2e-15</td> <td> 1.09e+14</td> <td> 0.000</td> <td>    1.000</td> <td>    1.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.540</td> <th>  Durbin-Watson:     </th> <td>   0.009</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.763</td> <th>  Jarque-Bera (JB):  </th> <td>   0.660</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.157</td> <th>  Prob(JB):          </th> <td>   0.719</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.757</td> <th>  Cond. No.          </th> <td>    761.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



I see that I get a perfect R^2.  So plugging this into my VIF formula 

$$ VIF = \frac{1}{1 - 1} $$ 

This is where the `inf` comes from.  I have a problem because I am able to perfectly predict `X1` from the other features.  Now lets see what happens when I remove the feature I created. 


```python
df.drop(['Y', 'X11'], axis = 1, inplace = True)
for i in range(1, 11):
    print(f'{df.columns[i]} {variance_inflation_factor(df.values, i)}')
```

    X1 1.1056812354246772
    X2 1.0741738574018993
    X3 1.0836168289787529
    X4 1.0410536144535685
    X5 1.0917593842615227
    X6 1.0955635908289256
    X7 1.0425296201253647
    X8 1.1118123228822059
    X9 1.051488347630868
    X10 1.1173223003485164


Now, I see that I get value for each of the 10 features.  We typically use a threshold of either below 5 or below 10 as a good VIF.  A VIF of 5 means that the other features are able to predict that feature with a R^2 of 0.80.  A VIF of 10 means that the other features are able to predict that feature with a R^2 of 0.90.  
