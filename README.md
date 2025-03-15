# Power Outage Exploration

By Kyle Flores

DSC80 Project

## Introduction
In this project, I analyzed a data set of major power outages witnessed by different states in the continental U.S. from January 2000 to July 2016. These major outages are defined by the Department of Energy to have had an effect on at least 50,000 customers, or causing an unplanned energy demand loss of at least 300 MegaWatts. This dataset is accessible throughPurdue University’s Laboratory for Advancing Sustainable Critical Infrastructure, at https://engineering.purdue.edu/LASCI/research-data/outages.

This dataset documents major power outages across various states in the US. In addition to outage events, it includes details on the geographical locations of outages, regional climate conditions, land-use characteristics, electricity consumption patterns, and the economic characteristics of affected states.

This first steps to this analysis include cleaning the dataset followed by exploratory data analysis to get acclimated to underlying patterns within the provided dataset. After these steps are performed, an analysis of the missingness mechanisms and dependency of the dataset will be performed.

Finally, I will explore my research question, which is how do aspects of a power outage affect the demand loss of electricity? I will attempt to build a model that predicts the peak or total electrical demand loss within a certain area. Finding answers to this question is important because it provides information about grid stability, which would help prevent widespread blackouts.

The original DataFrame contains 1534 rows and 57 columns to represent the 1534 outages that were documented as part of this dataset. However, for this project, the only columns to be used are as follows:

| Column                     | Description  |
|----------------------------|-------------|
| `YEAR`                     | Year of outage occurence  |
| `MONTH`                    | Month of outage occurence  |
| `U.S._STATE`               | State of outage occurence  |
| `NERC.REGION`              | North American Electric Reliability Corporation (NERC) regions of the outage event  |
| `CLIMATE.REGION`           | U.S. Climate regions defined by National Centers for Environmental Information (9 Regions)  |
| `ANOMALY.LEVEL`            | Oceanic El Niño/La Niña (ONI) index defining the cold and warm episodes by season  |
| `OUTAGE.START.DATE`        | Day of the year when outage event began  |
| `OUTAGE.START.TIME`        | Time of day when outage event began  |
| `OUTAGE.RESTORATION.DATE`  | Day of the year when power was restored to all customers  |
| `OUTAGE.RESTORATION.TIME`  | Time of day when power was restored to all customers  |
| `CAUSE.CATEGORY`           | Event categories causing the major power outages  |
| `OUTAGE.DURATION`          | Duration of outage events (minutes)  |
| `DEMAND.LOSS.MW`          | Peak electricity demand lost during outage event (Megawatt), Total Demand tends to be reported  |
| `CUSTOMERS.AFFECTED`       | Number of customers affected by outage event  |
| `TOTAL.PRICE`             | Average monthly electricity price in the U.S. state (cents/kilowatt-hour)  |
| `TOTAL.SALES`             | Total electricity consumption in the U.S. state (megawatt-hour)  |
| `TOTAL.CUSTOMERS`         | Annual number of total customers served in the U.S. state  |
| `POPPCT_URBAN`           | Percentage of the total population of the U.S. state represented by the urban population (in %)  |
| `POPDEN_URBAN`           | Population density of the urban areas (persons per square mile)  |
| `AREAPCT_URBAN`          | Percentage of the land area of the U.S. state represented by the land area of the urban areas (in %)  |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

To clean the data, several steps were performed. 

Step 1: Remove irrelevant rows that made up the first couple rows of the raw dataframe.  

Step 2: Define the columns for the dataframe using one the 4th row of the dataframe.  

Step 3: Combine OUTAGE.START.DATE with OUTAGE.START.TIME and OUTAGE.RESTORATION.DATE with OUTAGE.RESTORATION.TIME to create an OUTAGE.START and OUTAGE.RESTORATION column.  

Step 4: Select the following columns to be used for analysis: "YEAR", "MONTH", "U.S._STATE", "NERC.REGION", "CLIMATE.REGION", "ANOMALY.LEVEL", "CAUSE.CATEGORY", "OUTAGE.START", "OUTAGE.RESTORATION", "OUTAGE.DURATION", "DEMAND.LOSS.MW", "CUSTOMERS.AFFECTED",  "TOTAL.PRICE", "TOTAL.SALES", "TOTAL.CUSTOMERS",  "POPPCT_URBAN" "POPDEN_URBAN", "AREAPCT_URBAN".  

Step 5: Check the following columns for values of 0 and then replacing them with NaNs: 'OUTAGE.DURATION', 'CUSTOMERS.AFFECTED', 'DEMAND.LOSS.MW'.

The first few rows of the cleaned dataframe are as follows:

~~~
|   YEAR |   MONTH | U.S._STATE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CAUSE.CATEGORY     | OUTAGE.START        | OUTAGE.RESTORATION   |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   TOTAL.PRICE |   TOTAL.SALES |   TOTAL.CUSTOMERS |   POPPCT_URBAN |   POPDEN_URBAN |   AREAPCT_URBAN |
|-------:|--------:|:-------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:---------------------|------------------:|-----------------:|---------------------:|--------------:|--------------:|------------------:|---------------:|---------------:|----------------:|
|   2011 |       7 | Minnesota    | MRO           | East North Central |            -0.3 | severe weather     | 2011-07-01 17:00:00 | 2011-07-03 20:00:00  |              3060 |              nan |                70000 |          9.28 |       6562520 |           2595696 |          73.27 |           2279 |            2.14 |
|   2014 |       5 | Minnesota    | MRO           | East North Central |            -0.1 | intentional attack | 2014-05-11 18:38:00 | 2014-05-11 18:39:00  |                 1 |              nan |                  nan |          9.28 |       5284231 |           2640737 |          73.27 |           2279 |            2.14 |
|   2010 |      10 | Minnesota    | MRO           | East North Central |            -1.5 | severe weather     | 2010-10-26 20:00:00 | 2010-10-28 22:00:00  |              3000 |              nan |                70000 |          8.15 |       5222116 |           2586905 |          73.27 |           2279 |            2.14 |
|   2012 |       6 | Minnesota    | MRO           | East North Central |            -0.1 | severe weather     | 2012-06-19 04:30:00 | 2012-06-20 23:00:00  |              2550 |              nan |                68200 |          9.19 |       5787064 |           2606813 |          73.27 |           2279 |            2.14 |
|   2015 |       7 | Minnesota    | MRO           | East North Central |             1.2 | severe weather     | 2015-07-18 02:00:00 | 2015-07-19 07:00:00  |              1740 |              250 |               250000 |         10.43 |       5970339 |           2673531 |          73.27 |           2279 |            2.14 |

~~~

### Exploratory Data Analysis

#### Univariate Analysis

Data exploration will begin with univariate analysis to understand the distribution of a single variable at a time.

The first graph depicts the distribution of outages per cause category.

<iframe
  src="assets/category_counts.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Looking at the image, the most commonly reported cause for outage is severe weather, followed by intentional attack. The least commonly reported causes for outage are fuel supply emergency and islanding.

This next graph illustrates the amount of documented outages over time.

<iframe
  src="assets/outage_counts_over_time.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

As seen above, the amount of recorded outages spikes in the early 2010s, but returns to the previous levels in the later years.

#### Bivariate Analysis

The next steps of data exploration are bivariate analysis. Looking at these relationships provides information about how the different features interact and what relationships they have.

The following graph shows how the total amount of customers affected by outages changed over time.

<iframe
  src="assets/customers_affected_over_time.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The line graph above shows that certain periods have had very high amounts of customers affected, likely indicating that these periods had severe outages or many different outages. Most of the time periods tends to range about 500k to 2M customers affected by outages.

This scatter plot shows the relationship between outage duration and the loss of electricity demand.

<iframe
  src="assets/duration_vs_demand_loss.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

As shown above, there is no clear relationship between outage duration and the loss of electricity demand. A few outliers are present within the data but most of the data is clustered to have a demand loss of 0-1000 MW, no matter the duration.

#### Grouping and Aggregates

Grouping data gives information about patterns within groups of data. 

The following data is a grouping by climate region and aggregated by the mean of each feature:

| CLIMATE.REGION     |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |
|:-------------------|------------------:|-----------------:|---------------------:|
| West North Central |            796.07 |           391.2  |              66242.4 |
| Northwest          |           1536.36 |           343.93 |             148472   |
| Southwest          |           1621.41 |           909.76 |              87815.1 |
| West               |           1636.31 |           717.93 |             225303   |
| Southeast          |           2247.66 |           852.36 |             198593   |
| South              |           2872.45 |           471.65 |             206106   |
| Central            |           2882.21 |           574.36 |             144269   |
| Northeast          |           3330.52 |           981.36 |             175359   |
| East North Central |           5391.4  |           633.9  |             149816   |

Grouping by climate region informs us of the characteristics of each region. For example, the East North Central tends to have the longest outages while the West tends to have the most customers affected.

Other methods of displaying data characteristics is by a pivot table.

The pivot table below shows the mean amount of customers affected and outage duraton for each outage cause category.

| CAUSE.CATEGORY                |   CUSTOMERS.AFFECTED |   OUTAGE.DURATION |
|:------------------------------|---------------------:|------------------:|
| equipment failure             |            105451    |           1850.56 |
| fuel supply emergency         |                 1    |          13484    |
| intentional attack            |             18753.4  |            521.93 |
| islanding                     |              7232.72 |            200.55 |
| public appeal                 |             15999.4  |           1468.45 |
| severe weather                |            190972    |           3899.71 |
| system operability disruption |            211066    |            747.09 |

Looking at the pivot table above, we learn that system operability disruptions have the highest average amount of customers affected, while having shorter outage durations. Moreover, islandings affect less customers and have shorter durations.

## Assesment of Missingness

Many columns within the dataset contain missing values. However, a column that is likely to be NMAR is DEMAND.LOSS.MW because of the lack of necessity in reporting minor changes. The missingness could be due to very small or undetectable changes that were not recorded because the change was not noticeable, leading to missing values.

To determine if the data for DEMAND.LOSS.MW is MAR, additional data of the individual utility companies’ reporting policies to then perform missingness analysis and assess whether or not there is a correlation between certain policies and the amount of null values.

### Missingness Dependency

In order to assess the dependency of missingness, the distribution of duration will be explored against the missingness of cause and month.

#### Cause Missingness

Null Hypothesis: The distribution of cause category is the same when demand loss is missing vs not missing. 

Alternative Hypothesis: The distribution of cause category is different when demand loss is missing vs not missing. 

<iframe
  src="assets/cause_missingness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Plotting the distribution of cause category against demand loss missingness indicates that there is a difference in distributions. This is tested by a permutation test using TVD. The observed TVD was 0.44, which had a p-value of about 0. Because the p-value is below any significant level threshold that is commonly used, we reject the null hypothesis. There is significant evidence that the distribution of cause category is not the same when demand loss is missing vs not missing.

Below is the graph of the distribution of TVDs from the permutation tests.

<iframe
  src="assets/cause_missingness_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Month Missingness

Null Hypothesis: The distribution of month is the same when demand loss is missing vs not missing. 

Alternative Hypothesis: The distribution of month is different when demand loss is missing vs not missing.

<iframe
  src="assets/month_missingness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Plotting the distribution of month against demand loss missingness does not indicate a significant difference in distributions. This is tested by a permutation test using TVD. The observed TVD was 0.09, which had a p-value of about .0134. Because the p-value is above any significant level threshold that is commonly used, we fail to reject the null hypothesis. There is not significant evidence that the distribution of month is not the same when demand loss is missing vs not missing.

Below is the graph of the distribution of TVDs from the permutation tests.

<iframe
  src="assets/month_missingness_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Hypothesis Testing

Using a hypothesis test with a significance level of 0.05, I will test if the cause category of an outage has an effect on the electricity demand loss.

Null Hypothesis: On average, the demand loss from severe weather outages is the same as the duration of intentional attack outages.

Alternate Hypothesis: On average, the the demand loss of severe weather outages is greater than the duration of intentional attack outages.

Test Statistic: Difference in means will be used as the test statistic for this hypothesis test. The exact difference in means is severe weather - intentional attack.

The observed test statistic was 3377.78 MW. After conducting a permutation test with 10,000 simulations, the resulting p-value found was 0.0. Because this p-value is below the significant level threshold, we reject the null hypothesis. There is significant evidence that the demand loss from severe weather outages is not the same as the duration of intentional attack outages. Below is the distribution of the simulated difference in means.

<iframe
  src="assets/hypothesis_test_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Framing a Prediction Problem

My model will try to predict the amount of peak demand loss caused by an outage. This will be a regression problem because the target variable of peak demand loss is a continuous, quantitative variable. Predictor features will be used to influence the magnitude of demand loss.

To evaluate my model performance, I will use RMSE as the metric because it would be in the same units as the target variable, therefore easier to interpret.

At the time of prediction, we would have information about the state, NERC region, climate region, anomaly level, year, month, total sales, total price, total customers, and the urban features. This information can be utilized to predict the amount of peak demand loss caused by the outage.

## Baseline Model

My baseline model is a linear regression model that predicts the amount of peak demand loss by using Year, region information, as well as anomaly level to predict peak demand loss from the outages. This information is open for almost any company within the area and would be useful for determining the severity of outages and the necessity to fix the outage issue.

The feature of anomaly level is quantitative, year is ordinal, and NERC.REGION and CLIMATE.REGION are nominal. Anomaly level was chosen because it provides information about the climate that impacts whether or not the people demand electricity. Year was chosen because it reflects the changes that happen over time. NERC.REGION was chosen because it represents the energy usage policies in the region and CLIMATE.REGION indicates the pattern of demand loss that is typical within a certain region.

This baseline model had an RMSE of 1040 MW, which is not a good model performance because the mean demand loss is 704 MW, meaning that the predictions are extremely far away from the actual values.

## Final Model

The final model was a Ridge regression model that had the previous features with the addition of the ordinal month feature and the quantitative total.sales feature. Month was added as a predictor feature because it adds information about seasonal changes to demand that year does not quite capture. Total.sales was added because it would indicate the amount of electricity that is used during the time, which has a relationship with the amount of electricity that is demanded at any moment in time. 

Ridge regression was chosen because Ridge performs better than linear regression when there are more predictive features. 

The best hyperparameters found were found using GridSearchCv and are as follows::

Alpha: 300000  
Solver: Auto  
Polynomial Degree: 1  

The improved model had an RMSE of 950 MW, which is a 89 MW improvement from the baseline model. This is a large improvement, but still indicates that the model does not predict demand loss well.

## Fairness Analysis

Testing the fairness of the model will be done using month with a significance level of 0.05. The two groups will be the first half of the year (Months 1-6) and the second half of the year (Months 7-12). This feature was chosen because the month could 

Null Hypothesis: The model is fair. Its RMSE for the first half of the year and the second half of the year are roughly the same, and any differences are due to random chance. 

Alternative Hypothesis: Our model is unfair. Its RMSE for the first half of the year and the second half of the year are different.

Absolute difference in RMSE will be the test statistic of choice because it uses the metric used for model prediction and it avoids directional bias. It also indicates how different one model is from the other.

The observed absolute difference in RMSE was 194.11. This difference had a p-value of about 0.724, which is above the significance level, so we fail to reject the null hypothesis. There is not significant evidence that the model is unfair for different halves of the year. Below is the distribution of the test statistics.

<iframe
  src="assets/rmse_difference_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>