# UPPER CONFIDENCE BOUND

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
# declaring vars
d = 10 # no of ads
N = 10000 # no of visits
ads_selected = [] # vector for results
# step 1 - the num of times the ad i was selected up to round n
numbers_of_selections = [0] * d
# this creates a vector of size d, full of 0 values
# the sum of rewards of the ad i up to round n
sums_of_rewards = [0] * d
# vector again ^
total_reward = 0
# step 2.1 - average reward of ad i up to round n ri(n) == Ri(n)/Ni(n)
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
# step 2.2 - confidence interval |ri(n) - deltai (n), ri(n) + delta i (n)
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
# step 3 - select the ad i that has the max UCB = ri(n) + deltai (n)
# start at l. 25
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()