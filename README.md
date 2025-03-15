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

| Index | YEAR | MONTH | U.S._STATE | NERC.REGION | CLIMATE.REGION     | ANOMALY.LEVEL | CAUSE.CATEGORY      | OUTAGE.START         | OUTAGE.RESTORATION   | OUTAGE.DURATION | DEMAND.LOSS.MW | CUSTOMERS.AFFECTED | TOTAL.PRICE | TOTAL.SALES | TOTAL.CUSTOMERS | POPPCT_URBAN | POPDEN_URBAN | AREAPCT_URBAN |
|-------|------|-------|------------|-------------|---------------------|---------------|---------------------|----------------------|----------------------|----------------|---------------|-------------------|-------------|------------|----------------|--------------|--------------|--------------|
| 6     | 2011 | 7     | Minnesota  | MRO         | East North Central  | -0.3          | severe weather     | 2011-07-01 17:00:00  | 2011-07-03 20:00:00  | 3060.0         | NaN           | 70000.0           | 9.28        | 6562520    | 2595696        | 73.27        | 2279         | 2.14         |
| 7     | 2014 | 5     | Minnesota  | MRO         | East North Central  | -0.1          | intentional attack | 2014-05-11 18:38:00  | 2014-05-11 18:39:00  | 1.0            | NaN           | NaN               | 9.28        | 5284231    | 2640737        | 73.27        | 2279         | 2.14         |
| 8     | 2010 | 10    | Minnesota  | MRO         | East North Central  | -1.5          | severe weather     | 2010-10-26 20:00:00  | 2010-10-28 22:00:00  | 3000.0         | NaN           | 70000.0           | 8.15        | 5222116    | 2586905        | 73.27        | 2279         | 2.14         |
| 9     | 2012 | 6     | Minnesota  | MRO         | East North Central  | -0.1          | severe weather     | 2012-06-19 04:30:00  | 2012-06-20 23:00:00  | 2550.0         | NaN           | 68200.0           | 9.19        | 5787064    | 2606813        | 73.27        | 2279         | 2.14         |
| 10    | 2015 | 7     | Minnesota  | MRO         | East North Central  | 1.2           | severe weather     | 2015-07-18 02:00:00  | 2015-07-19 07:00:00  | 1740.0         | 250.0         | 250000.0          | 10.43       | 5970339    | 2673531        | 73.27        | 2279         | 2.14         |



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

