These are notes on using polynomial regression and step functions.  These are taken from 
- [Machine Learning 293 from Smith College by R. Jordan Crouser](http://www.science.smith.edu/~jcrouser/SDS293/labs/lab12-py.html)
- [Intro to Statistical Learning with Applications in R](http://faculty.marshall.usc.edu/gareth-james/ISL/)

#### Polynomial Regression

Polynomial regression is used to extend linear regression in which the relationship between your predictors and target is non-linear.  Below I am going to show an example of trying four different versions of polynomial regression. 


```python
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('https://raw.githubusercontent.com/sik-flow/datasets/master/Wage.csv')
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
      <th>year</th>
      <th>age</th>
      <th>maritl</th>
      <th>education</th>
      <th>region</th>
      <th>jobclass</th>
      <th>health</th>
      <th>health_ins</th>
      <th>logwage</th>
      <th>wage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006</td>
      <td>18</td>
      <td>1. Never Married</td>
      <td>1. &lt; HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>2. No</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004</td>
      <td>24</td>
      <td>1. Never Married</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>2. No</td>
      <td>4.255273</td>
      <td>70.476020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>45</td>
      <td>2. Married</td>
      <td>3. Some College</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.875061</td>
      <td>130.982177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>43</td>
      <td>2. Married</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>1. Yes</td>
      <td>5.041393</td>
      <td>154.685293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>50</td>
      <td>4. Divorced</td>
      <td>2. HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
  </tbody>
</table>
</div>



Apply polynomial regression to the age column 


```python
X1 = PolynomialFeatures(1).fit_transform(df.age.values.reshape(-1,1))
X2 = PolynomialFeatures(2).fit_transform(df.age.values.reshape(-1,1))
X3 = PolynomialFeatures(3).fit_transform(df.age.values.reshape(-1,1))
X4 = PolynomialFeatures(4).fit_transform(df.age.values.reshape(-1,1))
```

Fit a model using the age column 


```python
fit1 = sm.GLS(df.wage, X1).fit()
fit2 = sm.GLS(df.wage, X2).fit()
fit3 = sm.GLS(df.wage, X3).fit()
fit4 = sm.GLS(df.wage, X4).fit()


# Generate a sequence of age values spanning the range
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1)


# Predict the value of the generated ages
pred1 = fit1.predict(PolynomialFeatures(1).fit_transform(age_grid))
pred2 = fit2.predict(PolynomialFeatures(2).fit_transform(age_grid))
pred3 = fit3.predict(PolynomialFeatures(3).fit_transform(age_grid))
pred4 = fit4.predict(PolynomialFeatures(4).fit_transform(age_grid))
```

Plot out the model fits 


```python
fig, ax = plt.subplots(2,2, figsize = (12,5))

ax[0][0].scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3)
ax[0][0].plot(age_grid, pred1, color = 'b')
ax[0][0].set_ylim(ymin=0)
ax[0][0].set_title('Poly = 1')

ax[0][1].scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3)
ax[0][1].plot(age_grid, pred2, color = 'b')
ax[0][1].set_ylim(ymin=0)
ax[0][1].set_title('Poly = 2')

ax[1][0].scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3)
ax[1][0].plot(age_grid, pred3, color = 'b')
ax[1][0].set_ylim(ymin=0)
ax[1][0].set_title('Poly = 3')

ax[1][1].scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3)
ax[1][1].plot(age_grid, pred4, color = 'b')
ax[1][1].set_ylim(ymin=0)
ax[1][1].set_title('Poly = 4')

fig.subplots_adjust(hspace=.5)
```


![png](Polynomial_Regression_Step_Regression_files/Polynomial_Regression_Step_Regression_7_0.png)


### Step Functions 

Step functions can be used to fit different models to different parts of the data.  I am going to put the age column into 4 different bins. 


```python
df_cut, bins = pd.cut(df.age, 4, retbins = True, right = True)
df_cut.value_counts(sort = False)
```




    (17.938, 33.5]     750
    (33.5, 49.0]      1399
    (49.0, 64.5]       779
    (64.5, 80.0]        72
    Name: age, dtype: int64




```python
df_steps = pd.concat([df.age, df_cut, df.wage], keys = ['age','age_cuts','wage'], axis = 1)

# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_steps['age_cuts'])

# Statsmodels requires explicit adding of a constant (intercept)
df_steps_dummies = sm.add_constant(df_steps_dummies)

# Drop the (17.938, 33.5] category
df_steps_dummies = df_steps_dummies.drop(df_steps_dummies.columns[1], axis = 1)

df_steps_dummies.head(5)
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
      <th>const</th>
      <th>(33.5, 49.0]</th>
      <th>(49.0, 64.5]</th>
      <th>(64.5, 80.0]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fit3 = sm.GLM(df_steps.wage, df_steps_dummies).fit()
```


```python
# Put the test data in the same bins as the training data.
bin_mapping = np.digitize(age_grid.ravel(), bins)

# Get dummies, drop first dummy category, add constant
X_test2 = sm.add_constant(pd.get_dummies(bin_mapping).drop(1, axis = 1))

# Predict the value of the generated ages using the linear model
pred2 = fit3.predict(X_test2)

# Plot
fig, ax = plt.subplots(figsize = (12,5))
fig.suptitle('Piecewise Constant', fontsize = 14)

# Scatter plot with polynomial regression line
ax.scatter(df.age, df.wage, facecolor = 'None', edgecolor = 'k', alpha = 0.3)
ax.plot(age_grid, pred2, c = 'b')

ax.set_xlabel('age')
ax.set_ylabel('wage')
ax.set_ylim(ymin = 0);
```


![png](Polynomial_Regression_Step_Regression_files/Polynomial_Regression_Step_Regression_12_0.png)

