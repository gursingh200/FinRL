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


def compute_rewards(self,end_total_asset,begin_total_asset):
    if(self.reward_type == "Sharpe"):
        if(self.delta_reward == True):
            self.reward = rolling_sharpe_ratio(self,end_total_asset-begin_total_asset) - self.prev_reward
        else: 
            self.reward = rolling_sharpe_ratio(self,end_total_asset-begin_total_asset)

    elif(self.reward_type == "Sortino"):
        if(self.delta_reward == True):
            self.reward = rolling_downside_deviation_ratio(self,end_total_asset - begin_total_asset) - self.prev_reward
        else:
            self.reward = rolling_downside_deviation_ratio(self,end_total_asset - begin_total_asset)

    elif(self.reward_type == "Profit"):
        self.reward = end_total_asset - begin_total_asset
    else:
        print("Logical error in the naming")
        exit()