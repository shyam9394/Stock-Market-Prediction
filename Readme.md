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

![](https://lh3.googleusercontent.com/uPVOq-ownJyCwv1xi8Bu7I-oEyOtP06bacTOttwbHzR1dzH8nTTopU21h8ZYZMJBCJePlI03jOMCHvILdf7zun24snSEdBRGd__hrIfEzHVEmoiefZb5nWVUdfHjnGsBBNc2X0S2DcO-QsdX-ekZ8NrAQOc2tkOWdB8Vz6GFKS3t6ATzUpt7uaD5JRqP8W6vfBNxZW9rs-JjRTSepbzhaMkzYbO89KulpS9w5VaHJhrJP6g2LA9kqV8wBRxYeW-9tBrFetjeKi9u-auSRrlQXKu92VFUWseUlW4dRow9GpO1IHLbm4nJMHboYwq-eDmxFFiptKUhyR5mdPv1Jr6a7ugwGglyprO_HNaA7uDNsumQKtWUlFoiyo3-z63npNHOazgso6zz76AfK-CQgVrd45rVuZ8VNDZnPGb7b-tijckSJxI2bPYUHoPt-X3Z-PB-yZemOv3pYa8T5K3YxLyvTyRME8oo2gHPc-iCg9xdzdU6wcavp9a08JLauXupiRhIEvnIOFA0TTDZM40XOcfyqnJWqJBazaSWJoZff-yCycrnZxGSAQYCo1AXJirdZcgXKUYX5RxGlmtCmdCGcoRcaCS3cZDVE8T6YRolROYgCgccr8Vsp-bHBa28EGzRTM8=w586-h463-no)   

The stock data was taken from Kaggle. The dataset contains stock prices of S&P companies between 2014 and 2018. 

`stockdata.head()`

`stockdata.describe()`

### Section 1: Predicting ebay's stock price
![](https://i1.wp.com/mustbuyoffer.com/wp-content/uploads/2017/09/ebay3.jpg?fit=384%2C288&ssl=1)

#### Cleaning data

`ebay = stockdata.loc[stockdata["Name"]=="EBAY"]`    
![](https://lh3.googleusercontent.com/wbJPfTDbSxrMwDHyIHPe7urKqW9Df-Ekxo2cKfxgRk0WDzTAKdtAvJQ6UPyoZaPtxd7fvUvMGgGwjAREsA5D9RJxxJrVKaz1jy258cwZtI5lD27_KYupJQbAOyakkfGvwbH86jjU_h-S9mXfnaWkIRRXIOn9K5CqeASava-z8LQze7dyYwrqgeYx-CdYrURXBxqtY_U1jMOsYukibn7sx5Gs_XsV6nQ5h_JcetaxtVQ4KKqLJr5cZJTIBi85nMg8tpDlOQQ9Jr6whFljbAKDGpPsmkxDjas2SFUusSwX8Qhb-HWZCaDGTckbzn8z6vzKlKrRUI5eCiTxOlhDR93Xx3aMPBuZ077d-Gx5uzrbN8R9OTLu4wGzu4UdjamkK7mlOkhPdSEIgTgElx9GHSZBlfSLUI9GDFAGhp3K0o0iVKxQww91s2UsAfyy-JdhtVwtJ3EE_NTvTBGEd1wNb9dbZxw5rbkqz64LYmFbzH8ntV5d6Jot2g2MYVWDkDrN6HUzDogPg-C26u-uMjaCoBngBZLuh-DpdCcVoK5E82-Q2WoYrwXHpKX_4TAmQfMdhtHe_9xQVoPp3wOUNJW0tNG_5xdanbxnltJFTg1hKvE1KrP6i9khm9MQouVOjeE9rRI=w588-h459-no)

Creating a new dataset that contains only 'EBAY' stocks

`ebay.info()` 
  
![](https://lh3.googleusercontent.com/T4sajOiO2d4QNDdxdzyn2gBEZsUzcIc0tQ8DrytVZTy4lGGCNUfMYbLGF9KIX4FNlqO3fk0CRvgHN4MlXN7aVRmLVmXqs7dRcVl_01chj4ENV_L6a9m_EpyFZh7pbwHKsMZ2V_2l2cIeHX92uXDl1gPXPeApVsFbvvTSylW4HXjxLja8cD90qqLrz3pADv9NVNYltFHJddNb27e1ReCv9_LyNVHwRgdbLgPHZ0Dao0avJztLolqGNXQp6F-Ermu2iG1CAaNf7NyXQpbaKup6QqE_vQcRs7XrJo2HViWxKZSENnCMbTeb2UcJi9EY3XXf3V1knwjlCZAdO9BBJTiZYf8Ae8y9V-kTttaXFPazpCEBOXcCjxjGGSs8el1RVWAYacAoV90bKFsLrqrrch2Z0mbeqbf3Mg6UCIbMntYAiuBAv6xogWbMyH0WvPYLKOj0MyuTcSwRmLHRBIaYFh_s2mpv1Nb-jx6D3TTF59KEOkfnG4PsK2WIwAlMh8RoTJSQqvedAaJLzq_CvcGJIK4roLrEx56ORhzsvRvEhq9miJ9QSrau2hkDssGkLjTli0MtnOTLX6sWdQNqVqNZGIEgq2xsRplCeaLzmFqaj3vdSUwClMNHUz_1X112WxvMifo=w522-h220-no)   

Gives datatype of each column

`ebay.head()`

`ebay_df = ebay.copy()`   
Copying ebay dataframe to ebay_df dataframe to avoid setting warning

`ebay_df.loc[:,'date'] = pd.to_datetime(ebay_df.loc[:,'date'], format = '%Y/%m/%d')`   
Converting datatype of date to datetime 64

`ebay_df.info()`

![](https://lh3.googleusercontent.com/ipdrmd3Ohz6GBvSG-BPPh-np_BHFJ5MEXW9xQKQJJDI1jObo2wSJUKxj0YI7ZK3pKHvEb-j4NU5fjk5iiFR1c4jqz0-pUqDDVmaryEeg_AWVPPpGc53lv9ZRVR4ChWfqZ7q5QMXr7ZNIypWIl1lu4GQ4u6oEOT2vwHFbIKsakql7D525GxHi6DMYw58rZUI49SCGmdZcsRvdlnGjj2DM3xF-9Frb9xzwenT2_ma9caJOEO-Aa5kvNv3D4J5b-LX5jueqDi7fIKkJnLElYhgfBlUcq1UK-Gi9sd9YzHqltDfJsxvRglv343XcO8a4rGaTXlu1IOk5iQv-QISFiAdnhgp6vaqpnU01dkYIM80HI9cVEmepe5g640A28KHsDN4OdfnU5rHQmGUBswwiZ__DSlm8LrDTuMxWurCJ9noI7VPIgGndlgDJa4wulbI_VzIQOdmgkKxfU2i8fXuF9Chf33f062PmuOSlytCczYwTGkM8pdQiWYOwGYIYkX-w1L6u-w4hxfm-O10IXcLXAxjMkyi40hDgyru0TmXb8uyQTtDVS0Tsyxm24fevPI-p09lC_zKK0A4pg-g2aS_4mDXXr9tlIeUx94KYFEMN4JjTeGfjtWPPCtijy9wzVi_dq1o=w505-h241-no)

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

![](https://lh3.googleusercontent.com/UCR0Oy8W2TFz4zjNRQmkmHzgc2pQXV2Wgy2fiIRRMJomSZ04e1L3RuVDzjNEC53tA7BvqfFrVpNM3UIMf3wNv4wguJnZj99bbSvIL_UfGRU7q3I7IL2ImSgh1g2Xr_acAMEnZFOxGtrv0Qu4vz6QnoNyxtLUD25Dhey4MuiKVZs4tCum_WUcFV253LR5Jz6NE8T0oJChh9QVLl7HuEFClN5Ynz2V-EDON_ZcopkYvt-muHUjmSpnI2Yzj0OYdx5GH-26wa41mD9pRRQttbAtYAJBW9vCp64oR9bYJduPnHGxvcw9GHuwV_QFfEBO5CvjTR-8i909VMjxgFYtBYyepBcXGcz1YZn-Zr_MiQFBdl0UJcE6qmO-7b6DkRrfVGL3p0gjzXzrFOeIMuwp5x0LhHQa6dbF536BA69tskLoPV464k0Y0cnaLmQv_yjW501g0QpCS08ZnmY-doqWfKx5pR5GUhSDRiI4UL1bF8MSjv2y_Xytdr7C9lNBtLc8X6OY_dKieU5TYKOBoxKQtRE5FTeUhzGkVmG_UOrwksoETbB7Gz-2pmUY5xEJIO8dCr-UtllQmCb-EguAvg--YmiUyhXSRXApPcRaaxnjhVzAVkGRUtDMMn9HWSgKWt7H04I=w1400-h498-no)

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

![](https://lh3.googleusercontent.com/giye53--j_u8aofMK1VsvrSuGD31lETmomyvvClJrB-v9a7zOEzH_4NFcbmODFmE-SJwJVvR-8j_NgdJ4oP_sz-2CN_6DSQTpvK-XVU_O168BUwnq-h_c2oPyBHbm2AyKt0JrG8eApaRi_o2jqrUuvLIwgmP8M5FGJDnDRelEPatF3qsWVjPCZsrzE7i5dhUA7XAJVR2XstiorO3WHWdqppkwCpwKfGq4q4lvu-Jup7P6mrFayYbEOZjglcLVVhntWOBOEpW7K8GaJyEJE-S7SCPy-zIsoG982NzyDackypaBp9Q5pd-Seg4gmq0YRkfNVU5g6-rSmTObddQBKjd1S_TGqhster0vkMj3dfYnKsLhaE0HpDzdpnTO8HH4dz3cUXOjEVxlWF4zL9tc-YXzvld7dS96iX7C6tW3ucXOx3ANjMrA-Zect4t-roFjvzE_B36crwSm95WwG2EUCeAq34fGOZQ-YYv95e8EjnnqudM_WuVmE3m5_t3VNeSKaY-pww2hhNkqL6z6xbT_nOHK0Nbu738zhSJlo-ISnuedRofsNjkWFShLBBZdnaRpqAxugYIfm4Dr8F7EhuoPmj5R6kgBfQ3HmCUaqydMaqnfolwN_91K3tw6yMsHUQBcok=w1000-h598-no)  

Figure that contains trends and future prices of ebay stock

![](https://lh3.googleusercontent.com/eJN7n9Vgz9-ZgASYeqcykVDCe6vyuV-5JbWg2KKLF1dobThu-iLWoYKuudbeJgvH_AdDmW6bvUSIzIfe2p-ETQzg78-L0o6HqUnu9wKpJf66rOI9FtKaL5p-NDh8X-eyzuTUQvANLCyWJ9g-iC3SQ8LO3TmzIgiVDP-Sovbov0eaIKnpiyzO-jY9GQhT5GdVdJIW4GBBvhg33AbgXiv_cxRuVzYd8vJBBWZY7G5L8YGkBCidW8sKog2taMXgDFxjCedkxlEwL750Bsdg1pPnBKPdrrUWrbm6TlOutAL3jPSCjgjYHsMb8NlDWn6wbYM-MpoD4S91ZWbl4T8Zd1J-LENqBeQ5B_MlII7Dc_HcPGLjieZuLQ_6yYHAZzCyUFq9Q9z8C7_Y84KwiZKkkz3PLP3HNGSwVUa6q5XgxdpNMduoYNcaNwxfVsiaBxTSVjHU9TxP1PnN2K89h-PeWykhyXXXlXbXHJidLlsANX_zcHAQVMS2LNcsy4HFnUOu4kHElglcFS7nFB0v-l3Zl5nvfwjrRmd2MJyy0VrCdHPEc_UCQvdx6Sdp_h7lwzGEOcDpXzuwoiPM08usywPkVfurvtzLMt6CB0gjjY2OqwKgEIYcBlCWiisa40ikORNoC_Q=w891-h889-no)

Yearly, monthly and weekly trends.

#### ebay-Paypal split 

![](https://lh3.googleusercontent.com/MUScyW0AgfGYEnnmtSPplD0TaOSnUxeUf6hyPMjEcs-9TENLeCbyPMGzVJX0mMaeKkvM1bkvOO05suZT9bk4Njc8hd7gxSKQvMjUZYAi5j4Rc38_8m__8asJowvJ-ELX1inttghBf_82zx5tipnNrs6fMLBu5JXU8G5MbdZmDvT_-C2gGh1MNVxuJpPbTZqa3U5VYmAcpPh_TVh78qy73R5RWHbQKgXhttW5Mt3H_x7DWLTbkAo-K1FRu7VS0NvKpRjp4VX_oLcUsiyVRdc3yuWOGshjOOLOeCPujaWvUNgVQK5M6o91Kx7h4-ZNKzMzGGiVYYvApwL3xY0JwpWMT8aCErm5GvHl9xh0uKp1hsnMyT7Lk-REG33gB05zkV8ZUQqOp1gaJS5QwzdRv7OrjZvidy80aoZPocXkBBj9v8EsaTJPPJ3IAKBLqwN7NV6mA2asURhXX2qKzPDfQXNiRKhwlqxt9KNDKa0OyQGn8YqJ4qPEtM_A7JcNzkAqQlkGEjSmVitloD_6C29w-US9f_ZLQpQiCfLMuris38MsdMh6y9kqny6cIKfxjnvz3KVcIIL9Ur3re4NqwxGcuT4DzYGsW2ytdKEsV1f0OWlhPZ86X9F9iC2mEXKjRppA0Gs=w601-h472-no)

Paypal was split from its parent company ebay on July 17, 2015, that led to a huge drop in stock price.




### Section 2: Predicting Delta airlines' stock prices

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLDBIqGl7XVET25PvENv3sC0c2FuFS-riIDGbB5B5lcL4T2aqb)

#### Cleaning data

`delta = stockdata.loc[stockdata["Name"]=="DAL"]`

![](https://lh3.googleusercontent.com/Jm9ULP0cZHNSr-1tSLCX9qEgH87s2IGQ9WaOoF0m-0FnxAm52mxc565-8gMy56nR2lJ1QHt01Qh1R4-s6m2n9EQ1cSqMOVJByrofW7W6rOKnjeixiuuEYjab4yvl4h3k63iCY-YaKZsIKvV2R2SXSK6i596t5GWGkR77VPfgGwtoctPVjYBsbNi8yh7iQxhES09pURF27Dh0YvI0k2e5_atqjv5rLHRmG9dyrM1dazQda-YycB6RoISM6PT7jJf5XEBKw8Pi3A-YL5t954pSraYx9AgX6PmDvegKKJry95w8I0XwNDjq5mqTegiO6XeT_xYn7lbvKXTjdInPvlKHmRkJyTwofy2kBEk0fS4hiD-7NdVEoE15oKUt8jiHGaWN4EaMzkpeNvyga7al2RgPp6ZvPJMd3Si82JysqEbIu2QyL0tm0qJaOCszBnIhS8s_vYjuuqXadG6Y2hGFScq0L_dPHz_aQUqcnYbBVsbytwf1EfGr08TJPl0bO5yeEzMqXY2la8qwVQAJPlkdu_4uR6a_6_D8_HqsdIC2_sI_fi121B5xkyt9Lt1Ppqy-uWwy15J4LmDJDl3iVGidViW_br3VwFbaJJhnFXjEaHpnkYLPXOnwp91EDbWMwHp0LNY=w586-h459-no)

Creating a new dataset that contains only 'EBAY' stocks
`delta.info()`   

![](https://lh3.googleusercontent.com/Qj-bn2cs9TdGwWoSaiJWB-d7sr-nvGb-2m_wYsNi6L1B5Mspwqig96lNSDE-uKTOH5pMfI4vuOBUzyPr8Gz9JVb12RkH8-HqkrF0sJHhIvPu_pIiVgzrJuAenOmbQSNNlpD8gr7Nf012umMVZd3CyMdu7rjfePw-C9V5uSzJBAy8ehqGxh41HRWjTaVi6fybOXDy40D_tfHvb7K0LTSx-OtNJ9Px4eviX_YJ8GskZeb_OkwM_CAi8ZxMPtNse-G-i6rnWOkjepcgCs3pD9Cjx6iQqd1lzzMHvUFfrStyUXAUbET5EA72XmWs_lMytj7ICG9UjWQMB-66R6kurqIRq4zRcmrJ6cwbGT9Elvi-NsVvM9tPho51SBvXy4k6fWZ0eo0HUidxksSLjrh9FuLz1qUPsZ2tr6c4EvY6b4cyQV4kmujP5p6sMytWuGNX-g70tPGpJRYN1PKWZ4O7wu07brEf4yi1MqhqniHsp5Olon8Mr8Lvv81zkqRVLf6EcwPUvj0ekglucXTrc-HQRCXera1bnepVPjMUyM0xC4kmrCAw4ObTKYW7n3xlkYZ6Fm_AQgcd5_JVv6DBl63B6EBYWaFl_FSycguE6zIOOicjJK3TUE2OH2DzjH4p0KFfu4M=w486-h253-no)

`delta.head()`

`delta_df = delta.copy()`

`delta_df.loc[:,'date'] = pd.to_datetime(delta_df.loc[:,'date'], format = '%Y/%m/%d')`   
`delta_df.info()`

![](https://lh3.googleusercontent.com/VV7ukepeZRD1ERdKyaL15YxOPsNYgzxApf4c9JDCa2Y7R2uZjay00zdLXz_FDODUFQdOo157iwZi0R9x6LYeYnivvBVNlFaMDuQRnjktd-_Q_cQ5dcfGDGQe0c4Mv36cvRMYvc0pPSGHh4CtuSSfg2yvEyfk5oawsf8yyjkm0_fTPyowaUuzoFpdsgpkMrRqiE1sx-JovftY6ELZweBWhIJFzPz4SZSxcDpsy_hwoeWLbrmHtHIvZvK9loIDSfp8PYXmlF5hKlt7fo_CPIrHbF7-0Ftjc5NK1o8cltc5VQy_F7NcAcybgh0VBe-uesX73OHJKXWgS43UbrbUbYgQOCM9cb8ABXCdz1b8Dk-AbngsnJkl7Nk4jNfkJ9Bx6rGTThK5HolSxHvyhWyKJ8yPugGIzDHYSYrnSkHlFpu49AhIyqI3agyj7bZjalrIvQOpBmcfZ_NQTqkHdUaLu1-hHJ1_olOW5w4NKVNH-M62vg55rPDLEt2tScU26gWXO94_1TYBEQiUWZuTfqMNfULTbPnqYKgD2qoS7IVDZ66DrtOuVuKgRbxrGNCGmGcONDyAaGCNyQdhLyzB8A7Re3sinF6nHHtQ5QxYqaeKQDukzTscpK8beH-ZKgp1mAt39Nk=w612-h241-no)

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

![](https://lh3.googleusercontent.com/3BYqOoZKGOEAdJTi_oTD-6gvDTkArHC8cXg-97XIVGn_AGiv7dk_f7PHu8aUt5c_GvDgn1wBS6yxmGgOxugfQf95dECWKMRFCbH5LIEUGRbyPmTMks_5-eY33FUiNDt0lfnDjk_fjT4i0DbIhhIISzyJzjaxAqrf_xa2mr_dHYiE1dXd-vExJNtlhe91mPswfBLaQUbYCdNHVm6DOkAl1d2PQpwUJVm83KGI9KKdSL9h3zWrPUZ3Dohmk6YBx9soZsnMMNcv7SEYSvuGbBPpTbiCkRrTOrO1PuD42FwB7HYWPlkFu5gjoa63mocd7xnQvvtCi1zTjdmF4IIcapVSiUauPpgQBLv_7a7G-dlGsj18XKVi7DJOdsVIdIYNhZIcAp61VoosZ8KzxaD9UGhXDlrLRpdOEnPrM_SHbCi0Km89q__Sx9ohLBmKM9y2JtB6I22CD5IGXeGmPkfBB_hCvA6WK74PHlYv77Ggdwu0FcOLO4qieFUO1sIR-iAop0TIsd8QvAwLSxL79YalUI8G3F-ltrhOal24qLajhkskTcmzYGX5o__S5AD692vlOlhPfgm5f6gbJ83qF6nm88f1AWj_xR-cuXykM_vL4i8osdFoj1n7YXrP99QCVmAm-WE=w1400-h498-no)

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

![](https://lh3.googleusercontent.com/m7bnNc2qwI2XcPpnuPPL7ltHD2gXMoHhjSum2XB5IVoZQV2d4UdsAugrtg_ihOpddo1aXTSnz1VUZNnqohGhHIeGUmER4AY47gGeENrqWBSfE-Hs1I7DvCKrZtgC_jP1Q1gH1lLthMj4Vm8WMUa8fXFW8PX_EcAjNZKgVJrM9xjw7ZIFcaM2RCbsUDy99lDyg54WokDGf3UiSFCc-UYV26kDOQSGXy_rOH6sKdrJRgeviSGJzvN1nero5XDTDQkOhzwIFPTb6CR-VCJl7npEpu4RoB4gDF7FTv06sjlfY0SHmiDfACoZ7e61SWz4Ls03Cr8nEUubfA48khBnm_TmdvUw3RCpocZcUFwsc4bCgWkpvot2liE9T93-D0bu2POBp1GzFJJmhdzLt9WcgFozogL6q5RhE0gm8DBVEhK6z1nuSdHtISmKZs-ERKAlACCuLozHkt9h1WKDxrR6JTgRdUWkyxiTucq1-nbrgvGx35qmFzTbYmNYjZojk_P9Xoz6UZPWZdQ-UDtxSCVbOESRkxsAMq8eS1YhnnJGTwAhuQIMBS1zQePMDfArI9L8QiYlrWEU6q18H2WGt7nhENuZZJocc2gF2IEhAJPNCPTXgAbsgAa5HNokZ6JkRNZxyXU=w1000-h598-no)

`fig2 = model_d.plot_components(forecast_d)`  
`plt.show()`

![](https://lh3.googleusercontent.com/9c7q5uz4sw5Z_0fYJImJK6rzU7U2hnRDUAUrLwBzuM9nwjw_plFMdzxJEw1NlddBiSE4UL9mYq3WxzCE6zTBLYZF_rETvvnkgGmFuJSIcooUyU9u2jG_w9hHfagHpspJsM9gJkWKnC6RqdosPSv7gI-D-XDEA171PgyYszHvTDoxNg-OiMPUMVbd0Fvi2tfmXR2BqiK9IL-eLzE5k58-Vilq8bg849aM0ONaCBLLOOEbJY4eXvFXiy2GNJzJqAMIWCUcCNUUC6zGzNJ6E8Vmrg-IFGvWta965q8E8VtsuzvE5ebcAvuvZEwRxTAM7A8FMW5S9N-A08tyWTy8F6Br5Jvps9TKCGHna9XlxtkTwRCtsEqkTzZMMqnEQkSbsUe2OB7GQ7b_HxkFv_vk6LMTC3A0s2hoWi3CqEKyorA6zbdO6AHRQYLX_gkWmoTnzCL0YEcT0kuVNhx3Ki3ZlVusO09An5gsHP-F2St6S5dvPWvL8NKzuwyhDKEtqahM_TeLx03gysZU5j-zLhEHXPxwo5k4FFkMcIwkBMl41e79rBBoN9FSjPPKaJTZ-qI-xSfIQi7k76FnjtgpZV10yQRYxPgg6snAoIXsf4lz4zb-DCI3U0HepBR8l3oeM7qHIQE=w891-h889-no)

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






![](https://lh3.googleusercontent.com/QDAcv-eUVBJMgb27TgF68Hu-ltRHqJG-oYocJZNTFkLf0gh3Bl3ikIPkhvHlReKlqIGLdEb1lTTz4IWybRWQyXCRMel2Gp1787OGuRPmfz4BIfrVMXV_tm0elIkdWr1AGee2gkPyfZFQfpVk3N4sVotBpoS4_ZcUIGa6oxzp6Cn0wNf575Q7LjeW7AknSf32KjXhlFEtrRY9TdUZd9m6Ikef7LZlE9L6UauGS7elKuw-ptNRwUhd9saWna82EwT5NU8cYuD0VYk6PLHjeCPx4rnu-Ornf0Hbe_pgFedLZXncl8uYqlSFwnVqp7K4osC7os8lJoeL5dRJYEL6GZsoQEjLz2-m14HBNLqt38FociYbhHRkjgbygbD-Kp66pWGaXlVJXgyXh3laqu7G0ORq1yYP3F9g0nQmTXQAPVEi-mYK1dprMF6DfMnkZRF7eRf9MCe5cGScq2siGhBZrNQbTj29zNPjLk4HC5_VVQbCkodGhtyNu580DBuuA9P3ONOGR8-F9V_4WWemrIz9Rin-MeANEgcKazbeNQQd-m9kxWqaQBML5TWiFzVuhQkckIdLd35rNQMgdkbx5IW1L-O7uzZH8amIHttxdZZPPrTG2YwGMGYe9yCiVrP-nmrhH4w=w1800-h11400-no)







Various charts that signify seasonal trends, oil price change etc.










