import numpy as np
# import pyfolio as pyf
import pandas as pd
import pyfolio.plotting as pplot
import matplotlib.pyplot as plt
import os.path as path
import sys
result_path = path.join('results','overnight_run')
# data = pd.read_csv("results/account_value_trade_ensemble_126.csv")
# print(data)
# fig = pplot.plot_drawdown_underwater(data.account_value/1000)
# fig.plot()
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--window_size',type = int, default = 63)

parser.add_argument('--reward_type', default = 'Sortino', metavar = 'R',
                    help= 'One of Sortino, Sharpe, Profit')

parser.add_argument('--use_extra_features', default = True, type = bool)
                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--tr_s_date =', default= '2009-04-01',
                    help= 'Training start date (default: \'2009-04-01\')')
parser.add_argument('--tr_e_date =', default= '2021-01-01',
                    help= 'Training end date (default: \'2021-01-01\')')

parser.add_argument('--te_s_date =', default= '2021-01-01',
                    help= 'Testing start date (default: \'2021-01-01\')')
parser.add_argument('--te_e_date =', default= '2022-06-01',
                    help= 'Testing end date (default: \'2022-06-01\')')

parser.add_argument('--retrain_window',type = int, default=63)

args = parser.parse_args()

print(args)
# fig = data.plot()
# plt.show()
data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
        'Age': [20, 21, 19, 18]}
 
# Create DataFrame
df_account_value = pd.DataFrame(data)

df_account_value.to_csv("{}/{}_{}_{}_{}_{}.csv".format(result_path,args.reward_type,args.window_size,args.retrain_window,args.use_extra_features, args.seed))
print(result_path)
