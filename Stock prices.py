#Stock prices

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import seaborn as sb
from fbprophet import Prophet


import warnings
warnings.filterwarnings("ignore")


#White grid
plt.style.use("seaborn-whitegrid")


#Importing dataset
stockdata = pd.read_csv('Stockprices.csv')
stockdata.head()
stockdata.describe()


#Section 1: Ebay stock prediction

#Analyzing ebay's stocks
ebay = stockdata.loc[stockdata["Name"]=="EBAY"]
ebay.info()
ebay.head()

#Copying to a new dataframe to avoid setting warning
ebay_df = ebay.copy()
#Converting datetime object to datetime64 format

ebay_df.loc[:,'date'] = pd.to_datetime(ebay_df.loc[:,'date'], format = '%Y/%m/%d')
ebay_df.info()

#Plotting of ebay stock price 
#First plot(Close)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
ax1.plot(ebay_df['date'],ebay_df['close'])
ax1.set_xlabel('Date',fontsize = 12)
ax1.set_ylabel('Stock Price', fontsize=12)
ax1.set_title('Ebay Close Price History')
ax1.legend()

#Second subplot(High)
ax1.plot(ebay_df['date'],ebay_df['high'],color='green')
ax1.set_xlabel('Date',fontsize=12)
ax1.set_ylabel('Stock Price',fontsize = 12)
ax1.set_title('ebay High Price History')
ax1.legend()
#Third subplot(Low)
ax1.plot(ebay_df['date'],ebay_df['low'],color='red')
ax1.set_xlabel('Date',fontsize=12)
ax1.set_ylabel('Stock Price',fontsize=12)
ax1.set_title('ebay Low Price History')
ax1.legend()
#Fourth subplot(Volume)
ax2.plot(ebay_df['date'],ebay_df['volume'], color='purple')
ax2.set_xlabel('Date',fontsize=12)
ax2.set_ylabel('Stock Price', fontsize = 12)
ax2.set_title('ebay Volume History')
ax2.legend()


ppt_df=ebay_df.drop(['open','high','low','volume','Name'], axis =1)
ppt_df.rename(columns={'close':'y' , 'date':'ds'}, inplace=True)

#Using Prophet class from fbProphet library
model = Prophet()

model.fit(ppt_df)

#Creating Future dates
future_prices=model.make_future_dataframe(periods=365)

#Predicting prices
forecast = model.predict(future_prices)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

#Plotting future predictions
import matplotlib.dates as mdates

#Dates
starting_date = dt.datetime(2018, 2, 8)
starting_date1 = mdates.date2num(starting_date)
trend_date = dt.datetime(2017,12,4)
trend_date_1 = mdates.date2num(trend_date)

pointing_arrow = dt.datetime(2018, 2, 8)
pointing_arrow_1 = mdates.date2num(pointing_arrow)

fig = model.plot(forecast)
ax1 = fig.add_subplot(111)
ax1.set_title("ebay Stock Price Forecast", fontsize=16)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Close Price", fontsize=12)

# Forecast initialization arrow
ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow_1, 42), xytext=(starting_date1,44),
            arrowprops=dict(facecolor='#d2691e', shrink=0.1),
            )

# Trend emphasis arrow
ax1.annotate('Upward Trend', xy=(trend_date_1, 37), xytext=(trend_date_1,34),
            arrowprops=dict(facecolor='#39ff14', shrink=0.1),
            )

ax1.axhline(y=38, color='b', linestyle='-')

plt.show()

fig2 = model.plot_components(forecast)
plt.show()


#Section 2:Delta airlines stock prediction

delta = stockdata.loc[stockdata["Name"]=="DAL"]
delta.info()
delta.head()

#Copying to a new dataframe to avoid setting warning
delta_df = delta.copy()
#Converting datetime object to datetime64 format

delta_df.loc[:,'date'] = pd.to_datetime(delta_df.loc[:,'date'], format = '%Y/%m/%d')
delta_df.info()

#Plotting of delta airlines' stock price 
#First plot(Close)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
ax1.plot(delta_df['date'],delta_df['close'])
ax1.set_xlabel('Date',fontsize = 12)
ax1.set_ylabel('Stock Price', fontsize=12)
ax1.set_title('Delta Airlines Close Price History')
ax1.legend()

#Second subplot(High)
ax1.plot(delta_df['date'],delta_df['high'],color='green')
ax1.set_xlabel('Date',fontsize=12)
ax1.set_ylabel('Stock Price',fontsize = 12)
ax1.set_title('Delta Airlines High Price History')
ax1.legend()

#Third subplot(Low)
ax1.plot(delta_df['date'],delta_df['low'],color='red')
ax1.set_xlabel('Date',fontsize=12)
ax1.set_ylabel('Stock Price',fontsize=12)
ax1.set_title('Delta Airlines Low Price History')
ax1.legend()

#Fourth subplot(Volume)
ax2.plot(delta_df['date'],delta_df['volume'], color='purple')
ax2.set_xlabel('Date',fontsize=12)
ax2.set_ylabel('Stock Price', fontsize = 12)
ax2.set_title('Delta Airlines Volume History')
ax2.legend()



# Drop the columns
ppt_df_d = delta_df.drop(['open', 'high', 'low','volume', 'Name'], axis=1)
ppt_df_d.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

ppt_df_d.head()
model_d = Prophet()
model_d.fit(ppt_df_d)

future_prices_d = model_d.make_future_dataframe(periods=365)

# Predict Prices
forecast_d = model_d.predict(future_prices_d)
forecast_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Dates
starting_date_d = dt.datetime(2018, 2, 8)
starting_date_d1 = mdates.date2num(starting_date_d)
trend_date_d = dt.datetime(2017,10,11)
trend_date_d1 = mdates.date2num(trend_date_d)

pointing_arrow_d = dt.datetime(2018, 2, 8)
pointing_arrow_d1 = mdates.date2num(pointing_arrow_d)

fig_d = model_d.plot(forecast_d)
ax1 = fig_d.add_subplot(111)
ax1.set_title("Delta Airlines Stock Price Forecast", fontsize=16)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Close Price", fontsize=12)

# Forecast initialization arrow
ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow_d1, 57), xytext=(starting_date_d1,59),
            arrowprops=dict(facecolor='#d2691e', shrink=0.1),
            )

# Trend emphasis arrow
ax1.annotate('Upward Trend', xy=(trend_date_d1, 49), xytext=(trend_date_d1,44),
            arrowprops=dict(facecolor='#39ff14', shrink=0.1),
            )

ax1.axhline(y=50, color='b', linestyle='-')

plt.show()

fig2 = model_d.plot_components(forecast_d)
plt.show()
# Change dates from daily frequency to monthly frequency
forecast_monthly = forecast_d.resample('M', on='ds').mean()
forecast_monthly = forecast_monthly.reset_index() 


# Extract Year and Month and put it in a column.
forecast_monthly["month_int"] = forecast_monthly['ds'].dt.month
forecast_monthly["year"] = forecast_monthly['ds'].dt.year

forecast_monthly["month"] = np.nan
lst = [forecast_monthly]


for column in lst:
    column.loc[column["month_int"] == 1, "month"] = "January"
    column.loc[column["month_int"] == 2, "month"] = "February"
    column.loc[column["month_int"] == 3, "month"] = "March"
    column.loc[column["month_int"] == 4, "month"] = "April"
    column.loc[column["month_int"] == 5, "month"] = "May"
    column.loc[column["month_int"] == 6, "month"] = "June"
    column.loc[column["month_int"] == 7, "month"] = "July"
    column.loc[column["month_int"] == 8, "month"] = "August"
    column.loc[column["month_int"] == 9, "month"] = "September"
    column.loc[column["month_int"] == 10, "month"] = "October"
    column.loc[column["month_int"] == 11, "month"] = "November"
    column.loc[column["month_int"] == 12, "month"] = "December"
    
    
forecast_monthly['season'] = np.nan
lst2 = [forecast_monthly]

for column in lst2:
    column.loc[(column['month_int'] > 2) & (column['month_int'] <= 5), 'Season'] = 'Spring'
    column.loc[(column['month_int'] > 5) & (column['month_int'] <= 8), 'Season'] = 'Summer'
    column.loc[(column['month_int'] > 8) & (column['month_int'] <= 11), 'Season'] = 'Autumn'
    column.loc[column['month_int'] <= 2, 'Season'] = 'Winter'
    column.loc[column['month_int'] == 12, 'Season'] = 'Winter'
    
    

    

# Let's Create Seasonality Columns (Barplots that descripe the average trend per Season for each year)
# Create different axes by Year
df_2013 = forecast_monthly.loc[(forecast_monthly["year"] == 2013)]
df_2014 = forecast_monthly.loc[(forecast_monthly["year"] == 2014)]
df_2015 = forecast_monthly.loc[(forecast_monthly["year"] == 2015)]
df_2016 = forecast_monthly.loc[(forecast_monthly["year"] == 2016)]
df_2017 = forecast_monthly.loc[(forecast_monthly["year"] == 2017)]
df_2018 = forecast_monthly.loc[(forecast_monthly["year"] == 2018)]


f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,12))


# Year 2013
sb.pointplot(x="Season", y="trend",
                    data=df_2013, ax=ax1, color="g")

# Year 2014
sb.pointplot(x="Season", y="trend",
                    data=df_2014, ax=ax2, color="r")

# Year 2015
sb.pointplot(x="Season", y="trend",
                    data=df_2015, ax=ax3, color="r")


# Year 2016
sb.pointplot(x="Season", y="trend",
                    data=df_2016, ax=ax4, color="g")

# Year 2017
sb.pointplot(x="Season", y="trend",
                    data=df_2017, ax=ax5, color="g")

# Year 2018
sb.pointplot(x="Season", y="trend",
                    data=df_2018, ax=ax6, color="g")

ax1.set_title("Year 2013")
ax2.set_title("Year 2014")
ax3.set_title("Year 2015")
ax4.set_title("Year 2016")
ax5.set_title("Year 2017")
ax6.set_title("Year 2018")

# Oil dips
# September 2014 and June 2015
ax2.annotate('First Major \n Oil Price \n Decline \n(Starts Here)', xy=(3, 44), xytext=(2.8,38.5),
            arrowprops=dict(facecolor='#FE2E2E', shrink=0.1),
            )


ax3.annotate('Second \n Major \n Oil Price \n Decline \n(Starts Here)', xy=(1, 46.8), xytext=(1,44),
            arrowprops=dict(facecolor='#FE2E2E', shrink=0.1),
            )
