import pandas as pd
import numpy as np


def rolling_downside_deviation_ratio(self,cur_return):
    '''
    Function that computes the Sortino ratio.
    '''
    
    # rmv is the element we have to remove from the prev average and prev downside deviation
    rmv = self.returns_queue.pop(0)
    
    cur_average = self.prev_average + (cur_return-rmv)/self.window_size
    
    sq_downside_deviation = self.prev_deviation**2 + (np.minimum(cur_return,0)**2 - np.minimum(rmv,0)**2)/self.window_size
    
    downside_deviation = np.sqrt(sq_downside_deviation)
    
    if(downside_deviation!=0):
        reward = cur_average/downside_deviation
    else:
        reward = cur_average/0.01 # Assigning a very large reward to the agent, but not infinite  

    # Updating the class object for the next iteration of the loop
    self.returns_queue.append(cur_return)
    self.prev_average = cur_average
    self.prev_deviation = downside_deviation
    
    return reward

def rolling_sharpe_ratio(self,cur_return):
    '''
    Function that computes the Sharpe ratio
    '''
    # rmv is the element we have to remove from the prev average and prev downside deviation
    rmv = self.returns_queue.pop(0)
    
    cur_average = self.prev_average + (cur_return-rmv)/self.window_size
    
    window_variance = self.prev_deviation**2 + (cur_return**2 - rmv**2)/self.window_size
    
    std_deviation= np.sqrt(window_variance)
    
    if(std_deviation!=0):
        reward = cur_average/std_deviation
    else:
        reward = cur_average/0.01 # Assigning a very large reward to the agent, but not infinite  

    # Updating the class object for the next iteration of the loop
    self.returns_queue.append(cur_return)
    self.prev_average = cur_average
    self.prev_deviation = std_deviation
    
    return reward


def compute_vol(self):
    annual_factor = np.sqrt(252)
    numerator = self.price_dataset[:,1:] 
    denominator = self.price_dataset[:,:-1]
    return annual_factor*10*np.std(numerator/denominator,axis = 1)


def vol_adjustment(self,begin_state,end_state,upside= False,vol_baseline =1):


    vol_array = compute_vol(self)
    # print(self.price_dataset)
    # print(vol_array)
    begin_prices = begin_state[:self.stock_dim]
    end_prices = end_state[:self.stock_dim]
    begin_pos = begin_state[self.stock_dim:]
    end_pos = end_state[self.stock_dim:]


    # -------------- COMPUTATION OF RAW PROFITS ---------------------------------
    profit_arr = ((end_prices-begin_prices) * end_pos)


    # -------------- COMPUTATION OF VOL PROFITS ----------------------------------
    adjusted_vols = vol_array/vol_baseline

    if(upside == False):
        vol_adjusted_profits = profit_arr/adjusted_vols
        return np.sum(vol_adjusted_profits)

    # -------------- COMPUTATION OF UPSIDE VOL PROFITS ---------------------------
    else: 
        upside_vol_mask = profit_arr>0
        upside_vol = 1 + upside_vol_mask*(adjusted_vols-1)
        upside_vol_adjusted_profits = profit_arr/upside_vol
        return np.sum(upside_vol_adjusted_profits)

def compute_rewards(self,end_total_asset,begin_total_asset,begin_asset_state,end_asset_state):
    if(self.reward_type == "Sharpe"):
        cur_metric = rolling_sharpe_ratio(self,end_total_asset-begin_total_asset)
        if(self.delta_reward == True):
            self.reward = cur_metric - self.prev_metric
            self.prev_metric = cur_metric 
        else: 
            self.reward = cur_metric

    elif(self.reward_type == "Sortino"):

        cur_metric = rolling_downside_deviation_ratio(self,end_total_asset - begin_total_asset)
        if(self.delta_reward == True):
            self.reward = cur_metric - self.prev_metric
            self.prev_metric  = cur_metric
        else:
            self.reward = cur_metric
    # elif(self.reward_type == "Std_dev_profit"):
    #     self.reward = 
    elif(self.reward_type == "Profit"):
        self.reward = end_total_asset - begin_total_asset
    elif(self.reward_type == "VolProfit"):
        self.reward = vol_adjustment(self,begin_asset_state,end_asset_state,self.upside_vol)
    
    else:
        print("Logical error in the naming")
        exit()