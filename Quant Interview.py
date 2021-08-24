## Question 1 ##
## The information looks like its stored in SQL so I will provide SQL statements for the query.
## Part a ##
## Select e1.Name from
    ## Employee e1 join Employee e2 on e1.manager_id = e2.id
        ## where e1.Salary > e2.Salary

## Part b ##
## Select AVG(Salary) from Employee
    ## where id not in (SELECT UNIQUE(manager_id) from Employee)

## Question 2 ##
def exists(v):
    try:
        v
    except NameError:
        print("v is not defined")
    else:
        print("v is defined.")

## Question 3 ##
def pascal(n):
    lst = [1]
    for i in range(n):
        if i > 1:
            tmp = (i+1)*[1]
            for j in range(1, len(tmp)-1):
                tmp[j] = lst[j]+lst[j-1]
            lst = tmp
        else:
            lst = (i+1)*[1]
        print(" ".join(str(e) for e in lst))
    return

## Question 4 ##
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import warnings
warnings.filterwarnings("ignore")


stocks = ['AAPL','IBM','GOOG','BP','XOM','COST','GS']
stocks_rtns = ['AAPL log_return','IBM log_return','GOOG log_return','BP log_return','XOM log_return','COST log_return','GS log_return']
df = yf.download(  # or pdr.get_data_yahoo(... 
    # tickers list or string as well 
    tickers = stocks, 
    period = "6y", 
    interval = "1d", 
)
df = df[df.index.year == 2016]['Adj Close']

weights = np.array([0.15, 0.2, 0.2, 0.15, 0.1, 0.15, 0.05])
df['total'] = df[stocks].dot(weights)
df['log_return'] = df['total'].pct_change().apply(lambda x: np.log(1+x))
                                                                            
## Part a ##
## Calculate 1d Var in 2016 using the historical method
## Assumption: The Var95 and CVaR95 are in log returns terms.
VaR95 = df['log_return'].quantile(0.05)
CVaR95 = sum(filter(lambda x: x<VaR95, df['log_return']))/(len(df)*0.05)
print(VaR95) #-0.016694283935572048
print(CVaR95) #-0.025912370929775136

## Part b ##
## Calculate 1d Var in 2016 using parametric method
## Assumption: The covariance matrix, expected mean, Var95 and CVaR95 are in log returns term.
df[stocks_rtns] = df[stocks].pct_change().apply(lambda x: np.log(1+x))
cov_matrix = df[stocks_rtns].cov()

avg_rets = df[stocks_rtns].mean()
port_mean = avg_rets.dot(weights)
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
 
VaR95 = norm.ppf(0.05, port_mean, port_stdev)
CVaR95 = sum(filter(lambda x: x<VaR95, df['log_return']))/(len(df)*0.05)
print(VaR95) #-0.014322120365119826
print(CVaR95) #-0.03327831136879175

## Part c ##
## Rebalancing portfolio monthly by maximising the sharpe ratio of daily log returns for the previous month
## Assumption made: Risk free interest rate and transaction fee are unaccounted for.
## This rebalancing method works best when the returns and volatility are assumed remain the same for the following month.
## Usually daily returns in a 1 month period are insufficient to give a good representation of a stock expected volatility and returns, making monhtly rebalancing ineffective.
## Other rebalancing methods to consider are minimising volatility, maximising kelly criterion or maximising returns (momentum strategy)

months = []
monthly_weights = []
for month in range(1, 13):
    month_df = df[df.index.month == month]
    avg_rets = month_df[stocks_rtns].mean()
    cov_matrix = month_df[stocks_rtns].cov()

    num_portfolios = 25000
    #set up array to hold results
    #We have increased the size of the array to hold the weight values for each stock
    results = np.zeros((3+len(stocks),num_portfolios))

    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.random.uniform(low=-10, high=10, size=(len(stocks),)).astype("int")
        #rebalance weights to sum to 1
        if sum(abs(weights)) == 0:
            continue
        weights = weights/(math.log10(sum(abs(weights)))*10)
        weights = weights/sum(weights)
        
        #calculate portfolio return and volatility
        annual_port_mean = np.sum(avg_rets.dot(weights)) * 252
        annual_port_stdev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        #store results in results array
        results[0,i] = annual_port_mean
        results[1,i] = annual_port_stdev
        results[2,i] = results[0,i] / results[1,i]
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
            
    results = pd.DataFrame(results.T,columns=['ret','stdev','sharpe']+stocks)
    #locate position of portfolio with highest Sharpe Ratio
    new_weights = results.iloc[results['sharpe'].idxmax()][stocks]
    months.append(month)
    monthly_weights.append(new_weights)

monthly_end_weights = pd.DataFrame(monthly_weights)
monthly_end_weights.index = months
print(monthly_end_weights)
##             AAPL           IBM  ...          COST            GS
## 1  -0.000000e+00 -4.715284e+15  ... -0.000000e+00 -6.287046e+15
## 2  2.666667e+00  3.000000e+00  ...  1.000000e+00 -1.000000e+00
## 3   3.888889e-01  5.000000e-01  ...  3.333333e-01 -1.111111e-01
## 4  -4.149374e+15 -8.298747e+14  ... -6.638998e+15  1.659749e+15
## 5   4.285714e-01  3.571429e-01  ... -1.428571e-01 -2.857143e-01
## 6  -2.000000e+00 -0.000000e+00  ...  1.000000e+00 -1.500000e+00
## 7   2.000000e-01  9.000000e-01  ...  1.000000e-01 -0.000000e+00
## 8   2.000000e+00 -2.000000e+00  ... -2.000000e+00  5.000000e+00
## 9   5.000000e+00 -5.000000e+00  ... -6.000000e+00 -6.000000e+00
## 10  5.000000e+00 -1.000000e+00  ... -5.000000e+00  9.000000e+00
## 11  1.000000e+00  0.000000e+00  ... -8.333333e-01  1.333333e+00
## 12  5.454545e-01 -2.727273e-01  ... -0.000000e+00  2.727273e-01

## Question 5 ##
brew install cloc 
cd ~/my-python-project
## Part a and b ##
## CLOC can output number of files, total lines of blank, comment and code
cloc $(git ls-files)

## Part c ##
for branch in $(git branch):
    git grep -o "def" branch | wc -l
    
## Part d ##
git diff HEAD~3

## Part e ##
du -h --max-depth=2 /root/test

## Question 6 ##
import re

def count_dates(path):
    file = open(path, "r")
    data = file.read()

    count = 0
    ## YYYY/MM/DD
    parens1 = re.compile(r'([0-9]{4})\/([0][1-9]|[1][0-2])\/([0][1-9]|[1|2][0-9]|[3][0|1])')
    count += len(parens1.findall(data))
    ## MM/DD/YYYY
    parens2 = re.compile(r'([0][1-9]|[1][0-2])\/([0][1-9]|[1|2][0-9]|[3][0|1])\/([0-9]{4})')
    count += len(parens2.findall(data))
    ## DD/MM/YYYY
    parens3 = re.compile(r'([0][1-9]|[1|2][0-9]|[3][0|1])\/([0][1-9]|[1][0-2])\/([0-9]{4})')
    count += len(parens3.findall(data))
    ## MM/DD/YYYY or ## DD/MM/YYYY
    parens4 = re.compile(r'([0][1-9]|[1][0-2])\/([0][1-9]|[1][0-2])\/([0-9]{4})')
    count -= len(parens4.findall(data))
    ## DD (Jan/Feb/Mar/Apr/May/Jun/Jul/Aug/Sept/Oct/Nov/Dec) YYYY
    parens5 = re.compile(r'([0][1-9]|[1|2][0-9]|[3][0|1])\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)\s([0-9]{4})')
    count += len(parens5.findall(data))

    return count

