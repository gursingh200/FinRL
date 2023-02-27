
import numpy as np
import pandas as pd
# class Test:

#     def __init__(
#         self,
#         # returns_queue = list[float],
#         # prev_average = float,
#         # prev_downside_deviation = float,
#         window_size = int
#     ):
#         # Window size for the rolling sterling ratio
#         self.window_size = window_size
#         # Rewards queue to manage the rolling sterling ratio
#         self.returns_queue = [0]*self.window_size
#         self.prev_average = 0
#         self.prev_downside_deviation = 0 

        
#     def rolling_downside_deviation_ratio(self,cur_return):
            
#             # rmv is the element we have to remove from the prev average and prev downside deviation
#             rmv = self.returns_queue.pop(0)
            
#             cur_average = self.prev_average + (cur_return-rmv)/self.window_size
            
#             sq_downside_deviation = self.prev_downside_deviation**2 + (np.minimum(cur_return,0)**2 - np.minimum(rmv,0)**2)/self.window_size
#             # print(np.min(cur_return,0) ,np.min(rmv,0))
#             # print(min(cur_return,0))
#             downside_deviation = np.sqrt(sq_downside_deviation)
#             if(downside_deviation!=0):
#                 reward = cur_average/downside_deviation
#             else:
#                 reward = cur_average/0.01

#             # Updating the class object for the next iteration of the loop
#             self.returns_queue.append(cur_return)
#             self.prev_average = cur_average
#             self.prev_downside_deviation = downside_deviation
            
#             return reward

# t = np.random.randn(50)
# t = np.random.random(50)
# t = -t
# t = np.zeros(50)

# actions = np.array([0,1,2,-1,-2])
# print(t[0]+np.array(t[1:]))
# argsort_actions = np.argsort(actions)
# sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
# print(sell_index)
# print(test.returns_queue)
# for i in range(len(t)):
#     output_list.append([t[i],test.rolling_downside_deviation_ratio(t[i])])
# print(output_list)
# print(test.returns_queue)
# print(t)
# print("HEllo")
# test = Test(31)
# output_list = []
# t = [1,2,3,4,5,6]

# -------------------------------- TESTING VOLATILITY NORMALIZATION --------------------------------
def compute_vol(historical_prices):
    annual_factor = np.sqrt(252)
    numerator = historical_prices[:,1:] 
    denominator = historical_prices[:,:-1]
    return annual_factor*10*np.std(numerator/denominator,axis = 1)
    

stock_dim = 2
vol_window = 15
print(np.array([[0]*vol_window]*stock_dim))

begin_state = np.array([10,20,40,1, 1]) 
end_state = np.array([10,21,42,-1, 2]) 

historical_prices = np.array([
    [21,22,23,24,21,22,23,24,21,20,21,20,21,22,23],
    [41,42,43,44,41,42,43,44,41,42,42,43,44,41,42]
    ]) # YOU SHOULD INCLUDE CURRENT DAY PRICES AS WELL


new_prices = np.array([[40,42]])
print(np.concatenate((historical_prices,new_prices.T),axis = 1))

# vol_array = np.array([14,18])
vol_array = compute_vol(historical_prices)
vol_baseline = 1
sma_window = 3
window_size = 15


# compute_vol(historical_prices)



# --------------- EXTRACTION OF STATE DATA ----------------------------------
begin_acc = begin_state[0]
end_acc = end_state[0]
begin_prices = begin_state[1:1+stock_dim]
end_prices = end_state[1:1+stock_dim]
begin_pos = begin_state[1+stock_dim:]
end_pos = end_state[1+stock_dim:]


print("--------- Initial conditions ---------")
print("Initial account balance {}, Final account balance {}".format(begin_acc,end_acc))
print("Inital price {}, final price {}".format(begin_prices,end_prices))
print("Inital position {}, final position {}".format(begin_pos,end_pos))



# -------------- COMPUTATION OF RAW PROFITS ---------------------------------
profit = end_acc - begin_acc + (np.sum(end_prices * end_pos)) - (np.sum(begin_prices * begin_pos))
profit_arr = ((end_prices-begin_prices) * end_pos)


# -------------- COMPUTATION OF VOL PROFITS ----------------------------------
adjusted_vols = vol_array/vol_baseline
vol_adjusted_profits = profit_arr/adjusted_vols


# -------------- COMPUTATION OF UPSIDE VOL PROFITS ---------------------------
upside_vol_mask = profit_arr>0
upside_vol = 1 + upside_vol_mask*(adjusted_vols-1)
upside_vol_adjusted_profits = profit_arr/upside_vol
print(upside_vol_mask,upside_vol)



# -------------- PRINTING OF FINAL RESULTS -----------------------------------

print("-------------- Final results ---------------")

print("Raw profit", profit)
print("Assetwise profits",profit_arr)
print("---------------")
print("Adjusted vol array {}, \nVol adjusted profits {}, \nTotal vol profit {}".format(adjusted_vols,vol_adjusted_profits,np.sum(vol_adjusted_profits)))
print("---------------")
print("Upside vol array {}, \nUpside Vol profits {}, \nTotal upside vol profit {}".format(upside_vol, upside_vol_adjusted_profits,np.sum(upside_vol_adjusted_profits)))
print("---------------")











 
  
  
  
  
  
  
  
 

