High leverage points are points that have an unusual value for X.  It is important to identify these points because they can have a large impact on linear regression models.  

I am going to begin by creating a dataset. 


```python
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```


```python
X, y = make_regression(n_features = 1, noise=10, random_state=11)
```

I am adding a data point that has an unusual data point.  


```python
X = np.append(X, [3])
y = np.append(y, [70])

df = pd.DataFrame(X, columns=['X'])
df['Y'] = y
```

Plot my data and highlighting the unusual point as red.  


```python
plt.scatter(df['X'], df['Y'])
plt.scatter([3], [70], color = 'red')
plt.xlabel('X')
plt.ylabel('Y');
```


![png](Leverage_Points_files/Leverage_Points_6_0.png)


We see that the red point has the largest value of X and doesn't fit the general trend of the rest of the data.  I am going to use [Cook's Distance](https://en.wikipedia.org/wiki/Cook%27s_distance) to measure how much influence each point has.  Any point that has a large influence I want to be careful before I include it in my model. 


```python
from yellowbrick.regressor import CooksDistance

visualizer = CooksDistance()
visualizer.fit(df['X'].values.reshape(-1, 1), df['Y'])
```

    /Users/jeffreyherman/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.metrics.classification module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.metrics. Anything that cannot be imported from sklearn.metrics is now part of the private API.
      warnings.warn(message, FutureWarning)
    /Users/jeffreyherman/opt/anaconda3/lib/python3.7/site-packages/sklearn/base.py:213: FutureWarning: From version 0.24, get_params will raise an AttributeError if a parameter cannot be retrieved as an instance attribute. Previously it would return None.
      FutureWarning)
    /Users/jeffreyherman/opt/anaconda3/lib/python3.7/site-packages/yellowbrick/regressor/influence.py:183: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      self.distance_, linefmt=self.linefmt, markerfmt=self.markerfmt





    CooksDistance(ax=<matplotlib.axes._subplots.AxesSubplot object at 0x1c1f78ba50>)




![png](Leverage_Points_files/Leverage_Points_8_2.png)



```python
df['Distance'] = visualizer.distance_
df.sort_values('Distance', ascending = False).head()
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
      <th>X</th>
      <th>Y</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>3.000000</td>
      <td>70.000000</td>
      <td>3.658567</td>
    </tr>
    <tr>
      <th>30</th>
      <td>-2.653319</td>
      <td>-159.917692</td>
      <td>0.307768</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1.846365</td>
      <td>64.508669</td>
      <td>0.135441</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2.156674</td>
      <td>123.491855</td>
      <td>0.087477</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1.402771</td>
      <td>47.531992</td>
      <td>0.048779</td>
    </tr>
  </tbody>
</table>
</div>



We see that point 100 has a Cook's Distance that is the largest (typically any point with a Cook's Distance greater than 1 I will want to investigate). 

Lets see what happens to our regression when we keep a point that has high leverage. I am going to build 2 regression models - the first one will have the high leverage point and the second one will not have the high leverage point. 


```python
import statsmodels.api as sm

X = df['X']
Y = df['Y']

# coefficient 
X = sm.add_constant(X)

model_1 = sm.OLS(Y, X).fit()
model_1.summary()
```

    /Users/jeffreyherman/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.926</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.926</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1246.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 28 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>6.87e-58</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:56:18</td>     <th>  Log-Likelihood:    </th> <td> -407.59</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   101</td>      <th>  AIC:               </th> <td>   819.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    99</td>      <th>  BIC:               </th> <td>   824.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
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
  <th>const</th> <td>   -1.5736</td> <td>    1.377</td> <td>   -1.143</td> <td> 0.256</td> <td>   -4.306</td> <td>    1.159</td>
</tr>
<tr>
  <th>X</th>     <td>   49.8307</td> <td>    1.412</td> <td>   35.294</td> <td> 0.000</td> <td>   47.029</td> <td>   52.632</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>62.244</td> <th>  Durbin-Watson:     </th> <td>   1.371</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 403.422</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.867</td> <th>  Prob(JB):          </th> <td>2.50e-88</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>12.051</td> <th>  Cond. No.          </th> <td>    1.05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
from sklearn.metrics import mean_squared_error

mean_squared_error(df['Y'], model_1.predict(X))
```




    187.3842053021652




```python
X = df.drop(100)['X']
Y = df.drop(100)['Y']

# coefficient 
X = sm.add_constant(X)

model_2 = sm.OLS(Y, X).fit()
model_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>Y</td>        <th>  R-squared:         </th> <td>   0.952</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.951</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1933.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 28 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>2.59e-66</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:57:06</td>     <th>  Log-Likelihood:    </th> <td> -381.98</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   768.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    98</td>      <th>  BIC:               </th> <td>   773.2</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
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
  <th>const</th> <td>   -0.8252</td> <td>    1.115</td> <td>   -0.740</td> <td> 0.461</td> <td>   -3.037</td> <td>    1.386</td>
</tr>
<tr>
  <th>X</th>     <td>   52.5054</td> <td>    1.194</td> <td>   43.960</td> <td> 0.000</td> <td>   50.135</td> <td>   54.876</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 7.213</td> <th>  Durbin-Watson:     </th> <td>   1.615</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.027</td> <th>  Jarque-Bera (JB):  </th> <td>   9.901</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.307</td> <th>  Prob(JB):          </th> <td> 0.00708</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.414</td> <th>  Cond. No.          </th> <td>    1.07</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
mean_squared_error(df.drop(100)['Y'], model_2.predict(X))
```




    121.70966154143105



We see that the first model has a R^2 of 0.926 while the second model has a R^2 of 0.952.  The first model has a MSE of 194.9 and the second model has a MSE of 121.7.  Removing the high leverage improved both the R^2 and MSE. Now I am going to plot out both lines to see the impact. 


```python
Xnew = np.array([[1, df['X'].min()], [1, df['X'].max()]])
plt.scatter(df['X'], df['Y'])
plt.scatter([3], [70], color = 'red')
plt.plot([df['X'].min(), df['X'].max()], model_1.predict(Xnew), label = 'With Leverage Point', color = 'red')
plt.plot([df['X'].min(), df['X'].max()], model_2.predict(Xnew), label = 'Without Leverage Point')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1c2d5a9610>




![png](Leverage_Points_files/Leverage_Points_16_1.png)


The blue regression line (with the leverage point) is being pulled down towards that leverage point and is thus impacting the predictions for all the other data points. 
