
Below I am going to show the impact of making an interaction term.  

I am going to begin my loading in the credit dataset 


```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/sik-flow/datasets/master/Credit.csv', index_col=0)
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
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
    </tr>
    <tr>
      <td>2</td>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
    </tr>
    <tr>
      <td>3</td>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
    </tr>
    <tr>
      <td>4</td>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
    </tr>
    <tr>
      <td>5</td>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
    </tr>
  </tbody>
</table>
</div>



Make the `Student` column a dummy variable 


```python
df['Student'] = df['Student'].map(lambda x: 1 if x == 'Yes' else 0)
```

Use `Income` and `Student` to regress on `Balance` 


```python
import statsmodels.api as sm

X = df[['Income', 'Student']]
Y = df['Balance']

# coefficient 
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.277</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.274</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   76.22</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>9.64e-29</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:06:40</td>     <th>  Log-Likelihood:    </th> <td> -2954.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   5915.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   397</td>      <th>  BIC:               </th> <td>   5927.</td>
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
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>  211.1430</td> <td>   32.457</td> <td>    6.505</td> <td> 0.000</td> <td>  147.333</td> <td>  274.952</td>
</tr>
<tr>
  <th>Income</th>  <td>    5.9843</td> <td>    0.557</td> <td>   10.751</td> <td> 0.000</td> <td>    4.890</td> <td>    7.079</td>
</tr>
<tr>
  <th>Student</th> <td>  382.6705</td> <td>   65.311</td> <td>    5.859</td> <td> 0.000</td> <td>  254.272</td> <td>  511.069</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>119.719</td> <th>  Durbin-Watson:     </th> <td>   1.951</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>  23.617</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.252</td>  <th>  Prob(JB):          </th> <td>7.44e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.922</td>  <th>  Cond. No.          </th> <td>    192.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Now I am going to plot out the regression lines for when someone is a student and when someone is not a student 


```python
import numpy as np
import matplotlib.pyplot as plt

Xnew = np.array([[1, 0, 0], [1, 150, 0], [1, 0, 1], [1, 150, 1]])
preds = model.predict(Xnew)

plt.plot([0, 150], preds[:2], label = 'non-student')
plt.plot([0, 150], preds[2:], label = 'student')
plt.xlabel('income')
plt.ylabel('balance')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1c1dbafb70>




![png](Interaction_files/Interaction_7_1.png)


I see that the 2 lines are parallel.  The balance for students and non students both increase at the same rate as income increases.  But what if that is not the case?  That is where an interaction term comes into use.

I am going to make an interaction term between student and income. 


```python
df['Interaction'] = df['Student'] * df['Income']
```


```python
X = df[['Income', 'Student', 'Interaction']]
Y = df['Balance']

# coefficient 
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Balance</td>     <th>  R-squared:         </th> <td>   0.280</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.274</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   51.30</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>4.94e-28</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:07:38</td>     <th>  Log-Likelihood:    </th> <td> -2953.7</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   5915.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   396</td>      <th>  BIC:               </th> <td>   5931.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  200.6232</td> <td>   33.698</td> <td>    5.953</td> <td> 0.000</td> <td>  134.373</td> <td>  266.873</td>
</tr>
<tr>
  <th>Income</th>      <td>    6.2182</td> <td>    0.592</td> <td>   10.502</td> <td> 0.000</td> <td>    5.054</td> <td>    7.382</td>
</tr>
<tr>
  <th>Student</th>     <td>  476.6758</td> <td>  104.351</td> <td>    4.568</td> <td> 0.000</td> <td>  271.524</td> <td>  681.827</td>
</tr>
<tr>
  <th>Interaction</th> <td>   -1.9992</td> <td>    1.731</td> <td>   -1.155</td> <td> 0.249</td> <td>   -5.403</td> <td>    1.404</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>107.788</td> <th>  Durbin-Watson:     </th> <td>   1.952</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>  22.158</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.228</td>  <th>  Prob(JB):          </th> <td>1.54e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 1.941</td>  <th>  Cond. No.          </th> <td>    309.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
Xnew = np.array([[1, 0, 0, 0], [1, 150, 0, 0], [1, 0, 1, 0], [1, 150, 1, 150]])
preds = model.predict(Xnew)

plt.plot([0, 150], preds[:2], label = 'non-student')
plt.plot([0, 150], preds[2:], label = 'student')
plt.xlabel('income')
plt.ylabel('balance')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1c1dc9e4a8>




![png](Interaction_files/Interaction_11_1.png)


Now we see that the the 2 lines have different slopes.  The slope for students is lower than the slope for non-students.  This suggestions that incre
