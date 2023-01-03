
import numpy as np
import pandas as pd
class Test:

    def __init__(
        self,
        # returns_queue = list[float],
        # prev_average = float,
        # prev_downside_deviation = float,
        window_size = int
    ):
        # Window size for the rolling sterling ratio
        self.window_size = window_size
        # Rewards queue to manage the rolling sterling ratio
        self.returns_queue = [0]*self.window_size
        self.prev_average = 0
        self.prev_downside_deviation = 0 

        
    def rolling_downside_deviation_ratio(self,cur_return):
            
            # rmv is the element we have to remove from the prev average and prev downside deviation
            rmv = self.returns_queue.pop(0)
            
            cur_average = self.prev_average + (cur_return-rmv)/self.window_size
            
            sq_downside_deviation = self.prev_downside_deviation**2 + (np.minimum(cur_return,0)**2 - np.minimum(rmv,0)**2)/self.window_size
            # print(np.min(cur_return,0) ,np.min(rmv,0))
            # print(min(cur_return,0))
            downside_deviation = np.sqrt(sq_downside_deviation)
            if(downside_deviation!=0):
                reward = cur_average/downside_deviation
            else:
                reward = cur_average/0.01

            # Updating the class object for the next iteration of the loop
            self.returns_queue.append(cur_return)
            self.prev_average = cur_average
            self.prev_downside_deviation = downside_deviation
            
            return reward

# t = np.random.randn(50)
t = np.random.random(50)
# t = -t
# t = np.zeros(50)
test = Test(31)
output_list = []
# print(test.returns_queue)
for i in range(len(t)):
    output_list.append([t[i],test.rolling_downside_deviation_ratio(t[i])])
print(output_list)
# print(test.returns_queue)
# print(t)
# print("HEllo")