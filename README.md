# Pronsifier :
A Propensity Model helps to predict the likelihood that specific groups of customers will respond to a marketing campaign

## ABSTRACT

Companies in today's competitive market know they need strong strategies to attract customers. These strategies, built on data, are essential for success and revenue growth. Customer data helps businesses understand their audience, leading to smarter, data-driven marketing campaigns. However, these campaigns can be expensive, and if they don't work, companies lose money. That's why cost-effective approaches are crucial. This project demonstrates how a propensity model, using data science, can identify likely customers, offering a more efficient and successful way to market and get a good return on investment.

## GOAL

The goal is to create a model that predicts how likely individuals, leads, or customers are to respond to marketing campaigns. This model uses different statistical methods to analyze customer behavior and employs Machine Learning to estimate the probability that a customer will engage with a specific marketing strategy. Here's what the project involves:

- **Data Analysis**: This step involves getting the data ready, exploring it, and creating visualizations to understand it better.
- **Modeling**: This part is about building Machine Learning models to predict the chances of customers responding to marketing. It also includes visualizing and evaluating models to choose the best one.
- **Model Deployment Plan**: This step involves preparing the model for real-world use, making it reproducible, and ensuring it can work in a production setting.

## Architecture


![architecture](https://github.com/Sakshi1234-debug/pronsifier/assets/149681034/04dd3740-9bdb-4628-b5c6-e045f7dac09e)


## Data
There are 2 datasets provided. They are:
1. Train.csv which contains historical data of customers who have responded in 
yes/no.
2. Test.csv which contains a list of potential customers to whom to market. We 
will be predicting whether the customer will say yes/no on this data.

The column ‘responded’ will be used as the target variable to train the model. The 
corresponding value in the ‘test.csv’ will be ‘propensity’, which will be final predicted 
output.
From here on in this report, ‘train.csv’ will be called as ‘input data’

## Data Analysis
The columns and its corresponding inputs are:
![columns](https://github.com/Sakshi1234-debug/pronsifier/assets/149681034/1026a7f2-0818-4040-b855-253088c4e591)
As seen above, there are a lot of null values in various columns. We will be handling 
through methods like removing, imputing, etc.

## Data  Cleaning
### Remove Extra Columns:
We delete the 'id' and 'profit' columns since they aren't needed.
### Handle Missing Data:
1.For 'custAge', we replace missing values with the median age.

2.For 'schooling', we use the most common (mode) value to fill in gaps.

3.For 'day_of_week', we replace missing values with 'unknown'.

4.Adjust 'pdays' Column: In 'pdays', a value of 999 means it's the first call. We change these 999 values to '-1' to avoid treating them as outliers.






