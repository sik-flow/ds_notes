
I am going to show how to make a grouped bar chart and a 100% stacked bar chart.  To start I am going to load in a dataset.  I want to look at the relationship between number of customer service calls vs whether the customer churned or not. 


```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('churn.csv')
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
      <th>state</th>
      <th>account length</th>
      <th>area code</th>
      <th>phone number</th>
      <th>international plan</th>
      <th>voice mail plan</th>
      <th>number vmail messages</th>
      <th>total day minutes</th>
      <th>total day calls</th>
      <th>total day charge</th>
      <th>...</th>
      <th>total eve calls</th>
      <th>total eve charge</th>
      <th>total night minutes</th>
      <th>total night calls</th>
      <th>total night charge</th>
      <th>total intl minutes</th>
      <th>total intl calls</th>
      <th>total intl charge</th>
      <th>customer service calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.groupby('customer service calls')['churn'].value_counts()
```




    customer service calls  churn
    0                       False     605
                            True       92
    1                       False    1059
                            True      122
    2                       False     672
                            True       87
    3                       False     385
                            True       44
    4                       False      90
                            True       76
    5                       True       40
                            False      26
    6                       True       14
                            False       8
    7                       True        5
                            False       4
    8                       False       1
                            True        1
    9                       True        2
    Name: churn, dtype: int64



This is what I want to visualize.  I am first going to use a grouped bar chart. 

## Grouped Bar Chart 

#### Step 1: Make the above into a dataframe 


```python
# get value counts for non_churn calls 
non_churn = df[df['churn'] == False]['customer service calls'].value_counts().sort_index()
# rename to non_churn 
non_churn.rename('Non_Churn', inplace = True)
# get value counts for churn calls 
churn = df[df['churn'] == True]['customer service calls'].value_counts().sort_index()
# rename to churn
churn.rename('Churn', inplace = True)

# combine the 2 pandas series into a dataframe
churn_df = pd.concat([non_churn, churn], axis = 1)
churn_df
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
      <th>Non_Churn</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>605.0</td>
      <td>92</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1059.0</td>
      <td>122</td>
    </tr>
    <tr>
      <td>2</td>
      <td>672.0</td>
      <td>87</td>
    </tr>
    <tr>
      <td>3</td>
      <td>385.0</td>
      <td>44</td>
    </tr>
    <tr>
      <td>4</td>
      <td>90.0</td>
      <td>76</td>
    </tr>
    <tr>
      <td>5</td>
      <td>26.0</td>
      <td>40</td>
    </tr>
    <tr>
      <td>6</td>
      <td>8.0</td>
      <td>14</td>
    </tr>
    <tr>
      <td>7</td>
      <td>4.0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



I am going to fill in the missing values with 0, this is because the data is not missing rather no calls were made 


```python
churn_df.fillna(0, inplace = True)
churn_df
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
      <th>Non_Churn</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>605.0</td>
      <td>92</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1059.0</td>
      <td>122</td>
    </tr>
    <tr>
      <td>2</td>
      <td>672.0</td>
      <td>87</td>
    </tr>
    <tr>
      <td>3</td>
      <td>385.0</td>
      <td>44</td>
    </tr>
    <tr>
      <td>4</td>
      <td>90.0</td>
      <td>76</td>
    </tr>
    <tr>
      <td>5</td>
      <td>26.0</td>
      <td>40</td>
    </tr>
    <tr>
      <td>6</td>
      <td>8.0</td>
      <td>14</td>
    </tr>
    <tr>
      <td>7</td>
      <td>4.0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
churn_df.plot(kind = 'bar')
plt.xticks(rotation = 0)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), <a list of 10 Text xticklabel objects>)




![png](Charts_files/Charts_10_1.png)


#### Step 2: Make Plot 


```python
with plt.style.context('fivethirtyeight'):
    churn_df.plot(kind = 'bar', figsize = (8, 6))
    plt.xticks(rotation = 0)
    plt.xlabel('Number of Customer Service Calls')
    plt.ylabel('Count of Customers')
    plt.title('Number of Customer Service Calls - Churn vs No Churn')
```


![png](Charts_files/Charts_12_0.png)


#### Step 3: Interpret the Graph 

We see that when the number of customer service calls is between 0 and 3 there is a really high number of customers that did not churn.  However, after 3 customer service calls the number of instances is very small so it is hard to determine the impact after that.  

I am now going to show the same information with a 100% stacked bar chart.  Note I'll be using this [great tutorial](https://python-graph-gallery.com/13-percent-stacked-barplot/) 

## 100% Stacked Bar Chart 

#### Step 1: Make Plot 


```python
# From raw value to percentage
totals = [i+j for i,j in zip(churn_df['Non_Churn'], churn_df['Churn'])]
greenBars = [i / j * 100 for i,j in zip(churn_df['Non_Churn'], totals)]
orangeBars = [i / j * 100 for i,j in zip(churn_df['Churn'], totals)]

with plt.style.context('fivethirtyeight'):
    # plot
    barWidth = 0.85
    names = churn_df.index
    # Create green Bars
    plt.bar(range(len(churn_df)), greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label = 'No Churn')
    # Create orange Bars
    plt.bar(range(len(churn_df)), orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, 
           label = 'Churn')

    # Custom x axis
    plt.xticks(range(len(churn_df)), names)
    plt.xlabel("Number of Customer Service Calls")

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
```


![png](Charts_files/Charts_16_0.png)


#### Step 2: Interpret Graph 

In this plot we see that the percent of customers that churn after 4 customer service calls increases significantly.  My recommendation would be to prioritize customers that have made 2-3 customer service calls before they reach that 4 customer service calls threshold! 
