import pandas as pd
import numpy as np


def downside_deviation_lambda(data,window_size):
        """
        function to compute the downside deviation of an array
        :param data: (data) numpy array
        :return: (dd) float
        """
        return np.sqrt(np.sum(data[data<0]**2)/window_size)
    
def add_extra_features(df,reward_type,window_size):

    if(reward_type != "Profit"):        
        # Adding a feature which computes the returns over the window before so that it can be fed to the model. 
        # Padding with zeros in the start to ensure that there are no NaNs
        zero_data = np.zeros(shape=(window_size-1,len(df.columns)))
        pad = pd.DataFrame(zero_data, columns=df.columns)
        df = pd.concat([pad,df])
        
        # The ratios require the average return in the numerator
        df["rolling_avg"] = df["close"].rolling(window_size).sum()/window_size

        if(reward_type == "Sortino"):
            print("Adding Sortino ratio relevant features")
            # Adding the rolling downside deviation as the column for Sortino
            df["rolling_dd"] = df["close"].rolling(window_size).apply(lambda x: downside_deviation_lambda(x,window_size))

        if(reward_type == "Sharpe"):
            print("Adding Sharpe ratio relevant features")
            # Adding the rolling standard deviation as the column for Sharpe ratio
            df["rolling_stddev"] =df["close"].rolling(window_size).std()
            
        # Removing the added padding
        df = df.iloc[window_size-1:]
    else:
        
        print("No new features added as Profit is the reward")

    return df

data1 = [10,20,30,-40,50,60]
  
# Create the pandas DataFrame with column name is provided explicitly
df2 = pd.DataFrame(data1, columns=['close'])

print(add_extra_features(df2,"Sharpe",3))