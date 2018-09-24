# Stock Market Prediction
![](http://www.vir.com.vn/stores/news_dataimages/hung/012018/07/18/vns_ckvn2017.jpg)

## Introduction
Stock market is a very important part of economy of a country. Stocks have been one of the crucial factors that decided the fate of various organizations, ever since it was first in 1792. Many people invest on the stock market, so it is critical for them to buy/sell right stocks at the right time. 

Predicting stock prices using machine learning is an amazing option if people are unsure with choosing stocks. Prophet, an essential Facebook library is straight for approach predict future values of an element.

In this project, the stock prices of ebay and Delta Airlines have been predicted. There are several factors that affect stock prices, and some factors are discussed briefly.

## Prophet
Facebook's research team came with an easy implementation of forecasting using a powerful library called Prophet. The prophet blog states that an analyst rarely produces accurate forecasting data. This is one of the reasons why Facebook came up with an easier approach to solve complex problems that forecasts results. Prophet's team main goal is to to make it easier for experts and non-experts to make high quality forecasts that keep up with demand.

## Seaborn
Seaborn is a Python data visualization library based on matplotlib.It provides a high-level interface for drawing attractive and informative statistical graphics.



## Implementation
### Importing libraries

`import numpy as np`     
An essential library used for scientific computation.

`import matplotlib.pyplot as plt`    
Used for creating charts.

`import pandas as pd`
This library handles files.

`import datetime as dt`    
The datetime library contains classes for manipulating date and time in simple and complex ways.

`import seaborn as sb`    
Seaborn is a library based on matplotlib. It helps creating complex yet beautiful illustrations.

`from fbprophet import Prophet`    
The fbprophet is a library used for predicting future values.

`import warnings`    
This library is used to handle various warning.

`import matplotlib.dates as mdates` 


### Importing the dataset
`stockdata = pd.read_csv('Stockprices.csv')`    

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/2.jpg?raw=true)    

The stock data was taken from Kaggle. The dataset contains stock prices of S&P companies between 2014 and 2018. 

`stockdata.head()`

`stockdata.describe()`

### Section 1: Predicting ebay's stock price
![](https://i1.wp.com/mustbuyoffer.com/wp-content/uploads/2017/09/ebay3.jpg?fit=384%2C288&ssl=1)

#### Cleaning data

`ebay = stockdata.loc[stockdata["Name"]=="EBAY"]`    

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/3.jpg?raw=true)  
 
Creating a new dataset that contains only 'EBAY' stocks

`ebay.info()` 
  
![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/1.jpg?raw=true)  

Gives datatype of each column

`ebay.head()`

`ebay_df = ebay.copy()`   
Copying ebay dataframe to ebay_df dataframe to avoid setting warning

`ebay_df.loc[:,'date'] = pd.to_datetime(ebay_df.loc[:,'date'], format = '%Y/%m/%d')`   
Converting datatype of date to datetime 64

`ebay_df.info()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/4.jpg?raw=true)

#### Plotting of ebay stock price 

##### First plot(Close)
`f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))`   
`ax1.plot(ebay_df['date'],ebay_df['close'])`   
`ax1.set_xlabel('Date',fontsize = 12)`   
`ax1.set_ylabel('Stock Price', fontsize=12)`   
`ax1.set_title('Ebay Close Price History')`   
`ax1.legend()`

##### Second subplot(High)
`ax1.plot(ebay_df['date'],ebay_df['high'],color='green')`  
`ax1.set_xlabel('Date',fontsize=12)`   
`ax1.set_ylabel('Stock Price',fontsize = 12)`   
`ax1.set_title('ebay High Price History')`   
`ax1.legend()`

##### Third subplot(Low)
`ax1.plot(ebay_df['date'],ebay_df['low'],color='red')`   
`ax1.set_xlabel('Date',fontsize=12)`   
`ax1.set_ylabel('Stock Price',fontsize=12)`   
`ax1.set_title('ebay Low Price History')`   
`ax1.legend()`

##### Fourth subplot(Volume)
`ax2.plot(ebay_df['date'],ebay_df['volume'], color='purple')`   
`ax2.set_xlabel('Date',fontsize=12)`   
`ax2.set_ylabel('Stock Price', fontsize = 12)`   
`ax2.set_title('ebay Volume History')`   
`ax2.legend()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/Figure_1.png?raw=true)

Visualizing the close,low,high prices and volume of the present and past ebay stock.

##### Using Prophet class from fbProphet library

`ppt_df=ebay_df.drop(['open','high','low','volume','Name'], axis =1)`   
Creating a new dataframe ppt_df by dropping unnecessary columns. This is the dataframe on which Prophet is used.

`ppt_df.rename(columns={'close':'y' , 'date':'ds'}, inplace=True)`   
Renaming columns to plot charts with ease.

`model = Prophet()`   
`model.fit(ppt_df)`   
Creating an object(model) and fitting it on the dataframe (ppt_df).

`future_prices=model.make_future_dataframe(periods=365)`   
Creating 365 future dates in addition to the existing dates on the dateframe (ppt_df) using make_future_dataframe method.

`forecast = model.predict(future_prices)`    
 Creating a dataframe that contains the predicted prices.

`forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()`

##### Plotting future predictions  
`starting_date = dt.datetime(2018, 2, 8)`   
`starting_date1 = mdates.date2num(starting_date)`   
`trend_date = dt.datetime(2017,12,4)`   
`trend_date_1 = mdates.date2num(trend_date)`

`pointing_arrow = dt.datetime(2018, 2, 8)`   
`pointing_arrow_1 = mdates.date2num(pointing_arrow)`

`fig = model.plot(forecast)`   
`ax1 = fig.add_subplot(111)`  
`ax1.set_title("ebay Stock Price Forecast", fontsize=16)` 
`ax1.set_xlabel("Date", fontsize=12)`   
`ax1.set_ylabel("Close Price", fontsize=12)`

`ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow_1, 42), xytext=(starting_date1,44),`
            `arrowprops=dict(facecolor='#d2691e', shrink=0.1))`

`ax1.annotate('Upward Trend', xy=(trend_date_1, 37), xytext=(trend_date_1,34),`
            `arrowprops=dict(facecolor='#39ff14', shrink=0.1),)`

`ax1.axhline(y=38, color='b', linestyle='-')`   
`plt.show()`

`fig2 = model.plot_components(forecast)`   
`plt.show()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/Figure_2-1.png?raw=true)

Figure that contains trends and future prices of ebay stock

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/Figure_2.png?raw=true)

Yearly, monthly and weekly trends.

#### ebay-Paypal split 

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/july.png?raw=true)

Paypal was split from its parent company ebay on July 17, 2015, that led to a huge drop in stock price.




### Section 2: Predicting Delta airlines' stock prices

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLDBIqGl7XVET25PvENv3sC0c2FuFS-riIDGbB5B5lcL4T2aqb)

#### Cleaning data

`delta = stockdata.loc[stockdata["Name"]=="DAL"]`

![](https://lh3.googleusercontent.com/Jm9ULP0cZHNSr-1tSLCX9qEgH87s2IGQ9WaOoF0m-0FnxAm52mxc565-8gMy56nR2lJ1QHt01Qh1R4-s6m2n9EQ1cSqMOVJByrofW7W6rOKnjeixiuuEYjab4yvl4h3k63iCY-YaKZsIKvV2R2SXSK6i596t5GWGkR77VPfgGwtoctPVjYBsbNi8yh7iQxhES09pURF27Dh0YvI0k2e5_atqjv5rLHRmG9dyrM1dazQda-YycB6RoISM6PT7jJf5XEBKw8Pi3A-YL5t954pSraYx9AgX6PmDvegKKJry95w8I0XwNDjq5mqTegiO6XeT_xYn7lbvKXTjdInPvlKHmRkJyTwofy2kBEk0fS4hiD-7NdVEoE15oKUt8jiHGaWN4EaMzkpeNvyga7al2RgPp6ZvPJMd3Si82JysqEbIu2QyL0tm0qJaOCszBnIhS8s_vYjuuqXadG6Y2hGFScq0L_dPHz_aQUqcnYbBVsbytwf1EfGr08TJPl0bO5yeEzMqXY2la8qwVQAJPlkdu_4uR6a_6_D8_HqsdIC2_sI_fi121B5xkyt9Lt1Ppqy-uWwy15J4LmDJDl3iVGidViW_br3VwFbaJJhnFXjEaHpnkYLPXOnwp91EDbWMwHp0LNY=w586-h459-no)

Creating a new dataset that contains only 'DAL' stocks   
`delta.info()`   

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/del-1.jpg?raw=true)

`delta.head()`

`delta_df = delta.copy()`

`delta_df.loc[:,'date'] = pd.to_datetime(delta_df.loc[:,'date'], format = '%Y/%m/%d')`   
`delta_df.info()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/del-2.jpg?raw=true)

Gives datatype of each column


`delta_df = delta.copy()`   
Copying delta dataframe to delta_df dataframe to avoid setting warning

`delta_df.loc[:,'date'] = pd.to_datetime(delta_df.loc[:,'date'], format = '%Y/%m/%d')`   
Converting datatype of date to datetime 64

#### Plotting of delta airlines' stock price   
##### First plot(Close)  
`f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))`  
`ax1.plot(delta_df['date'],delta_df['close'])`  
`ax1.set_xlabel('Date',fontsize = 12)`  
`ax1.set_ylabel('Stock Price', fontsize=12)`  
`ax1.set_title('Delta Airlines Close Price History')`


##### Second subplot(High)  
`ax1.plot(delta_df['date'],delta_df['high'],color='green')`  
`ax1.set_xlabel('Date',fontsize=12)`  
`ax1.set_ylabel('Stock Price',fontsize = 12)`  
`ax1.set_title('Delta Airlines High Price History')`

##### Third subplot(Low)  
`ax1.plot(delta_df['date'],delta_df['low'],color='red')`  
`ax1.set_xlabel('Date',fontsize=12)`  
`ax1.set_ylabel('Stock Price',fontsize=12)`  
`ax1.set_title('Delta Airlines Low Price History')`

##### Fourth subplot(Volume)  
`ax2.plot(delta_df['date'],delta_df['volume'], color='purple')`  
`ax2.set_xlabel('Date',fontsize=12)`  
`ax2.set_ylabel('Stock Price', fontsize = 12)`  
`ax2.set_title('Delta Airlines Volume History')`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/del-4.png?raw=true)

Visualizing the close,low,high prices and volume of the present and past ebay stock.

##### Using Prophet class from fbProphet library

`ppt_df_d=delta_df.drop(['open','high','low','volume','Name'], axis =1)`   
Creating a new dataframe ppt_df by dropping unnecessary columns. This is the dataframe on which Prophet is used.

`ppt_df_d.rename(columns={'close':'y' , 'date':'ds'}, inplace=True)`   
Renaming columns to plot charts with ease.

`model_d = Prophet()`   
`model_d.fit(ppt_df_d)`   
Creating an object(model) and fitting it on the dataframe (ppt_df_d).

`future_prices_d=model.make_future_dataframe(periods=365)`   
Creating 365 future dates in addition to the existing dates on the dateframe (ppt_df_d) using make_future_dataframe method.

`forecast_d = model.predict(future_prices_d)`    
 Creating a dataframe that contains the predicted prices.

`forecast_d[['ds','yhat','yhat_lower','yhat_upper']].tail()`

##### Plotting future predictions 
 
`starting_date_d = dt.datetime(2018, 2, 8)`  
`starting_date_d1 = mdates.date2num(starting_date_d)`  
`trend_date_d = dt.datetime(2017,10,11)`  
`trend_date_d1 = mdates.date2num(trend_date_d)`

`pointing_arrow_d = dt.datetime(2018, 2, 8)`  
`pointing_arrow_d1 = mdates.date2num(pointing_arrow_d)`

`fig_d = model_d.plot(forecast_d)`  
`ax1 = fig_d.add_subplot(111)`  
`ax1.set_title("ebay Stock Price Forecast", fontsize=16)`  
`ax1.set_xlabel("Date", fontsize=12)`  
`ax1.set_ylabel("Close Price", fontsize=12)`

`ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow_d1, 57), xytext=(starting_date_d1,59),`
            `arrowprops=dict(facecolor='#d2691e', shrink=0.1),)`

`ax1.annotate('Upward Trend', xy=(trend_date_d1, 49), xytext=(trend_date_d1,44),`
            `arrowprops=dict(facecolor='#39ff14', shrink=0.1),)`

`ax1.axhline(y=50, color='b', linestyle='-')`
`plt.show()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/del_5.png?raw=true)

`fig2 = model_d.plot_components(forecast_d)`  
`plt.show()`

![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/del_6.png?raw=true)

`forecast_monthly = forecast_d.resample('M', on='ds').mean()`  
`forecast_monthly = forecast_monthly.reset_index() `
Changing date from daily frequency to monthly frequency.
 
`forecast_monthly["month_int"] = forecast_monthly['ds'].dt.month`  
`forecast_monthly["year"] = forecast_monthly['ds'].dt.year`  
Extracting month and year and putting it in the same column.

`forecast_monthly["month"] = np.nan` 
`lst = [forecast_monthly]`


`for column in lst:`     
   `column.loc[column["month_int"] == 1, "month"] = "January"`  
   `column.loc[column["month_int"] == 2, "month"] = "February"`  
   `column.loc[column["month_int"] == 3, "month"] = "March"`  
   `column.loc[column["month_int"] == 4, "month"] = "April"`  
   `column.loc[column["month_int"] == 5, "month"] = "May"`  
   `column.loc[column["month_int"] == 6, "month"] = "June"`  
   `column.loc[column["month_int"] == 7, "month"] = "July"`  
   `column.loc[column["month_int"] == 8, "month"] = "August`"  
   `column.loc[column["month_int"] == 9, "month"] = "September"`  
   `column.loc[column["month_int"] == 10, "month"] = "October"`   
   `column.loc[column["month_int"] == 11, "month"] = "November"`  
   `column.loc[column["month_int"] == 12, "month"] = "December"`
    
    
`forecast_monthly['season'] = np.nan`  
`lst2 = [forecast_monthly]`

`for column in lst2:`   
    `column.loc[(column['month_int'] > 2) & (column['month_int'] <= 5), 'Season'] = 'Spring' `
  
    `column.loc[(column['month_int'] > 5) & (column['month_int'] <= 8), 'Season'] = 'Summer'` 
  
    `column.loc[(column['month_int'] > 8) & (column['month_int'] <= 11), 'Season'] = 'Autumn'` 

    `column.loc[column['month_int'] <= 2, 'Season'] = 'Winter'`  
    `column.loc[column['month_int'] == 12, 'Season'] = 'Winter'`
    
    

    

#### Creating Seasonality Columns (Barplots that describe the average trend per Season for each year)
##### Create different axes by Year

`df_2013 = forecast_monthly.loc[(forecast_monthly["year"] == 2013)]`  
`df_2014 = forecast_monthly.loc[(forecast_monthly["year"] == 2014)]`  
`df_2015 = forecast_monthly.loc[(forecast_monthly["year"] == 2015)]`  
`df_2016 = forecast_monthly.loc[(forecast_monthly["year"] == 2016)]`  
`df_2017 = forecast_monthly.loc[(forecast_monthly["year"] == 2017)]`  
`df_2018 = forecast_monthly.loc[(forecast_monthly["year"] == 2018)]`


`f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,12))`


##### Year 2013
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2013, ax=ax1, color="g")`

##### Year 2014
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2014, ax=ax2, color="r")`

##### Year 2015
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2015, ax=ax3, color="r")`


##### Year 2016
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2016, ax=ax4, color="g")`

##### Year 2017
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2017, ax=ax5, color="g")`

##### Year 2018
`sb.pointplot(x="Season", y="trend",`
                    `data=df_2018, ax=ax6, color="g")`

`ax1.set_title("Year 2013")`  
`ax2.set_title("Year 2014")`  
`ax3.set_title("Year 2015")`  
`ax4.set_title("Year 2016")`  
`ax5.set_title("Year 2017")`  
`ax6.set_title("Year 2018")`

#### Oil dips
##### September 2014 and June 2015
`ax2.annotate('First Major \n Oil Price \n Decline \n(Starts Here)', xy=(3, 44), xytext=(2.8,38.5),`
            `arrowprops=dict(facecolor='#FE2E2E', shrink=0.1),)`


`ax3.annotate('Second \n Major \n Oil Price \n Decline \n(Starts Here)', xy=(1, 46.8), xytext=(1,44),`
            `arrowprops=dict(facecolor='#FE2E2E', shrink=0.1),)`






![](https://github.com/shyam9394/Stock-Market-Prediction/blob/master/Stock%20images/Figure_4.png?raw=true)





Various charts that signify seasonal trends, oil price change etc.










