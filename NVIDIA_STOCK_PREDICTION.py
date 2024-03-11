#!/usr/bin/env python
# coding: utf-8
## Stock market prediction using python and machine learning. LSTM method
STAGE 1: EXPLORATORY DATA ANALYSIS

# In[68]:


#import the necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# For time stamps
from datetime import datetime


# In[39]:


# Define the ticker symbol
ticker_symbol = 'NVDA'

# Set the start and end dates for the data
start_date = '2014-01-31'
end_date = '2024-01-31'

# Download the data
nvidia_data = yf.download(ticker_symbol, start=start_date, end=end_date)


# In[38]:


print(nvidia_data.head())
print(nvidia_data.tail())


# In[50]:


#checking for missing values
nvidia_data.isna().sum()
#getting the information on the data types
nvidia_data.info()


# In[63]:


#exploring the data further to find the mean, std, percentiles, max and min values to understand the volatility of the data.
nvidia_data.describe()


# From the previous code that uses the describe() function to better understand the volatility of the data, we come up with the following findings.
# 1. The mean values of Open, High, Low, Close are close to each other which indicates relatively stable pricing.
# 2. The standard deviations of the prices are quite high, which shows considerable variability or volatility in the prices.
# 3. The median ('50%') values are closer to the 25th percentile ('25%') than the 75th percentile ('75%'), indicating a slight skew towards lower prices.
# 4. There is considerable standard deviation in the trading volume, which also shows that there is volatility of the price in the shares. High standard deviation in both the prices and the volume indicates that the market for this asset experiences significant fluctuations and volatility.
# 5. The mean of the 30-day moving average is 102.99 and the standard deviation for the moving average is relatively high, suggesting variability in the trend of the closing prices over the observed period.The moving average varies from a minimum of around 4.39 to a maximum of approximately 537.39.

# In[46]:


#plotting the close price against the dates to see the movement of the stock over 10 years.
nvidia_data['Open'].plot(figsize = (16, 10))


# In[54]:


#getting the 7-day rolling mean, which is the moving average of the past 30 days
nvidia_data.rolling(7).mean().head(20)


# In[56]:


#comparing the plots of the rolling mean and the entire data set plot
nvidia_data['Open'].plot(figsize = (16, 6))
nvidia_data.rolling(window=30).mean()['Close'].plot()


# In[61]:


#plotting the close column versus the 30 day moving average
nvidia_data['Close: 30 Day Mean'] = nvidia_data['Close'].rolling(window=30).mean()
nvidia_data[['Close', 'Close: 30 Day Mean']].plot(figsize = (16, 6))


# In[62]:


#specifying the number of periods as 1
nvidia_data['Close'].expanding(min_periods = 1).mean().plot(figsize = (16, 6))


# In[65]:


#histograms that give us more information on the distribution of the data by comparing the original data and the normal distribution data
mean = nvidia_data['Close'].mean()
std_dev = nvidia_data['Close'].std()
num_samples = len(nvidia_data)
normal_distribution = np.random.normal(mean, std_dev, num_samples)
plt.figure(figsize=(10, 6))
plt.hist(nvidia_data['Close'], bins = 20, alpha = 0.5, label ='Original Data')
plt.hist(normal_distribution, bins = 20, alpha = 0.5, label = 'Normal Distribution Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Comparison of Original data and normal distributed data')
plt.legend()
plt.grid(True)
plt.show()


# In[71]:


#To further understand the structure of the data, we print the skewness and the kurtosis of the normal_data
print(skew(normal_distribution))
print(kurtosis(normal_distribution))


# In[73]:


#to understand the structure of the distribution, print the skewness and kurtosis of the original nvidia_data
print(skew(nvidia_data['Close']))
print(kurtosis(nvidia_data['Close']))


# 1. From the skewness and kurtosis data, we can see that the data is positively skewed meaning that majority of the data points are concentrated on the left side. In financial terms, this could suggest that there are more instances of small fluctuations in the stock price, with fewer occurrences of large positive movements.
# 2. When it comes to kurtosis, a value greater than 3 indicates heavy tails compared to a normal distribution. This suggests that the distribution has more extreme values (both positive and negative) than a normal distribution. In financial terms, this could indicate that the stock price experiences more extreme fluctuations than would be expected under a normal distribution.
# 3. Based on the skewness and kurtosis values provided, it seems that the data is likely following a leptokurtic distribution, which means it has heavier tails and a higher peak compared to a normal distribution. This is often observed in financial data, where extreme events (such as large price movements) are more common than what a normal distribution would predict.
# 
# 
# 
# 
# 
# 

# In[75]:


#we perform a normalty test on normal_distribution data
from scipy.stats import shapiro

# Perform Shapiro-Wilk test for normality
statistic, p_value = shapiro(normal_distribution)

# Print the test statistic and p-value
print("Shapiro-Wilk Test Statistic:", statistic)
print("p-value:", p_value)

# Interpret the result based on the p-value
alpha = 0.05
if p_value > alpha:
    print("Data looks normally distributed (fail to reject H0)")
else:
    print("Data does not look normally distributed (reject H0)")


# In[77]:


#we perform the normality test on the original data nvidia stock price

statistic, p_value = shapiro(nvidia_data['Close'])
#print the statistic and the p_value
print("Shapiro-Wilk Test Stastic:", statistic)
print("p_value:", p_value)
#interpret the results according to the p_value
alpha = 0.05
if p_value > alpha:
    print("Data looks normally distributed (fail to reject H0)")
else:
    print("Data does not look normally distributed (reject H0)")


# In[ ]:




