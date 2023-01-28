from curses import window
from turtle import down
import pandas as pd
import numpy as np
import random
tmp  = pd.DataFrame(np.random.randn(2000,2)/10000, 
                    index=pd.date_range('2001-01-01',periods=2000),
                    columns=['open','close'])

print(tmp)

def downside_deviation_lambda(data,window_size):
    """
    function to compute the downside deviation of an array
    :param data: (data) numpy array
    :return: (dd) float
    """
    return np.sqrt(np.sum(data[data<0]**2)/window_size)
        

def add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """

        reward_type ="Sharpe"
        window_size = 5
        df = data.copy()
        # df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)

        if(reward_type != "Profit"):
            
            # Adding a feature which computes the returns over the window before so that it can be fed to the model. 
            # Padding with zeros in the start to ensure that there are no NaNs
            zero_data = np.zeros(shape=(window_size-1,len(df.columns)))
            pad = pd.DataFrame(zero_data, columns=df.columns)
            df = pd.concat([pad,df])
            
            # The ratios require the average return in the numerator
            df["rolling_avg"] = df["close"].rolling(window_size).sum()/window_size

            if(reward_type == "Sortino"):
                # Adding the rolling downside deviation as the column for Sortino
                df["rolling_dd"] = df["close"].rolling(window_size).apply(lambda x: downside_deviation_lambda(x,window_size))

            if(reward_type == "Sharpe"):
                # Adding the rolling standard deviation as the column for Sharpe ratio
                df["rolling_stddev"] =df["close"].rolling(window_size).std()
                
            # Removing the added padding
            df = df.iloc[window_size-1:]
        
        return df

tmp = add_user_defined_feature(tmp)
print(tmp)