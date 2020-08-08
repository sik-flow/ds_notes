In this post, I am going to show how the signs can change as we add more features to a logistic regression equation.  I am also going to explain why the signs flip.  

To start I am loading in the `Default` dataset. 


```python
import pandas as pd 

df = pd.read_csv('https://raw.githubusercontent.com/sik-flow/datasets/master/Default.csv')
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
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>



I am first going to build a model using student (a categorical feature) to predict whether someone defaults on a loan or not.  


```python
import statsmodels.api as sm

X = df[['student']]
y = df['default']

X = sm.add_constant(X)

logit = sm.Logit(y, X).fit()
logit.summary()
```

    /Users/jeffreyherman/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm


    Optimization terminated successfully.
             Current function value: 0.145434
             Iterations 7





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>default</td>     <th>  No. Observations:  </th>  <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  9998</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 01 Aug 2020</td> <th>  Pseudo R-squ.:     </th> <td>0.004097</td> 
</tr>
<tr>
  <th>Time:</th>                <td>21:12:23</td>     <th>  Log-Likelihood:    </th> <td> -1454.3</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>0.0005416</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>   -3.5041</td> <td>    0.071</td> <td>  -49.554</td> <td> 0.000</td> <td>   -3.643</td> <td>   -3.366</td>
</tr>
<tr>
  <th>student</th> <td>    0.4049</td> <td>    0.115</td> <td>    3.520</td> <td> 0.000</td> <td>    0.179</td> <td>    0.630</td>
</tr>
</table>



I see that the coefficient for student is positive which means that a student has a probability of defaulting than a non-student I can check this with the following. 


```python
import numpy as np

# probabiliy of student 
np.exp(-3.5041 + 0.4049) / (1 + np.exp(-3.5041 + 0.4049))
```




    0.04314026622102699




```python
# probabiliy of non-student 
np.exp(-3.5041) / (1 + np.exp(-3.5041))
```




    0.029195798210381152



We see the probability of a student to default is 4.3% and a non-student is 2.9% from our logistic regression model.  This should line up with what our original data is.  We can check this with the following: 


```python
# student 
df[df['student'] == 1]['default'].value_counts(normalize = True)
```




    0    0.956861
    1    0.043139
    Name: default, dtype: float64




```python
# non-student 
df[df['student'] == 0]['default'].value_counts(normalize = True)
```




    0    0.970805
    1    0.029195
    Name: default, dtype: float64



We see that it does line up with our logistic regression model.  Now lets see what happens when I add an additional feature `balance` to the model. 


```python
X = df[['student', 'balance']]
y = df['default']

X = sm.add_constant(X)

logit = sm.Logit(y, X).fit()
logit.summary()
```

    Optimization terminated successfully.
             Current function value: 0.078584
             Iterations 10





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>default</td>     <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9997</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 01 Aug 2020</td> <th>  Pseudo R-squ.:     </th>   <td>0.4619</td>  
</tr>
<tr>
  <th>Time:</th>                <td>21:12:23</td>     <th>  Log-Likelihood:    </th>  <td> -785.84</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>1.189e-293</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>  -10.7495</td> <td>    0.369</td> <td>  -29.115</td> <td> 0.000</td> <td>  -11.473</td> <td>  -10.026</td>
</tr>
<tr>
  <th>student</th> <td>   -0.7149</td> <td>    0.148</td> <td>   -4.846</td> <td> 0.000</td> <td>   -1.004</td> <td>   -0.426</td>
</tr>
<tr>
  <th>balance</th> <td>    0.0057</td> <td>    0.000</td> <td>   24.748</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.15 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.



We now see that student has a negative coefficient, which means that students when balance is held constant students default at a lower rate than non-students.  This contradicts what our single variable logistic regression models says.  Lets take a look at why this happens. 

I am first going to build a model that shows the rate at which students and non-students default at different balances. 


```python
import seaborn as sns 
import matplotlib.pyplot as plt
```


```python
my_vals = [0] + [i for i in range(500, 2500, 100)] + [3000]
student = []
non_student = []
for counter, x in enumerate(my_vals):
    if counter + 1 < len(my_vals):
        stud_val = df[(df['student']==1) & (df['balance'] > x) & (df['balance'] < my_vals[counter+1])]['default']
        student.append((stud_val == 1).sum() / len(stud_val))
        nstud_val = df[(df['student']==0) & (df['balance'] > x) & (df['balance'] < my_vals[counter+1])]['default']
        non_student.append((nstud_val == 1).sum() / len(nstud_val))
```


```python
plt.figure(figsize = (12, 8))
plt.plot(range(500, 2600, 100), non_student, color = 'red', label = 'non-student')
plt.plot(range(500, 2600, 100), student, color = 'blue', label = 'student')
plt.axhline(df[df['student'] == 0]['default'].value_counts(normalize = True)[1], 
           color = 'red', ls = '--')
plt.axhline(df[df['student'] == 1]['default'].value_counts(normalize = True)[1], 
           color = 'blue', ls = '--')
plt.legend(fontsize = 16)
plt.xlabel('Credit Card Balance', fontsize = 18)
plt.ylabel('Default Rate', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16);
```


![png](Logistic%20Regression%20_files/Logistic%20Regression%20_15_0.png)


The horizontal lines are the overall default rate for students (blue) and non-students.  We see that the blue dotted line is above the red dotted line, this is why when looking at the single variable logistic regression it showed that students were more likely to default than non-students.  The line plot shows that at the same credit card balance, students are more likely to not default than non-students. 

Additionally, I want to look at the balance distribution for students and non-students. 


```python
plt.figure(figsize = (12, 8))
sns.boxplot(x = 'student', y = 'balance', data = df, palette =  ['r', 'b'])
plt.xticks(range(2), ['non-student', 'student'], fontsize = 16)
plt.xlabel('')
plt.ylabel('Credit Card Balance', fontsize = 18)
plt.yticks(fontsize = 16);
```


![png](Logistic%20Regression%20_files/Logistic%20Regression%20_17_0.png)


We see that students on average have a higher balance than non-students.  But, we see when a student and a non-student have the same balance - the student is less likely to default. 

Lets look at the probability of default for a student and non-student when they both have a $2000 balance. 


```python
# student
np.exp(-10.75 + 0.0057 * 2000 - 0.7149) / (1 + np.exp(-10.75 + 0.0057 * 2000 - 0.7149))
```




    0.483780692590808




```python
# non-student
np.exp(-10.75 + 0.0057 * 2000) / (1 + np.exp(-10.75 + 0.0057 * 2000))
```




    0.6570104626734988



A student has a probability of defaulting at 48% and a non-student has a probability of defaulting of 66%.  

#### References
- [An Introduction to Statistical Learning with Applications in R](http://faculty.marshall.usc.edu/gareth-james/ISL/)
