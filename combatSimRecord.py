import os
'''
This section simply records the latest simulation being saved as 
a recording
'''
def writeLog(filepath,count):
    file = open(filepath,"w+")
    file.write(str(count))
    file.close()

def readLog(filepath):
    count = int([line for line in open(filepath,"r")][0])
    return count

import gym
import ma_gym
from ma_gym.wrappers import Monitor

count = readLog(os.getcwd()+'/count.txt')
print(type(count),":",count)

env = gym.make('Combat-v0')
target = int(input("How many simulations would you like to perform?"))

for i in range(target):
    count += 1
    print(count)
    directory = f'ma-gym/recordings/{count}'
    env = Monitor(env, directory=directory)
    env.reset()

    #obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    obs_e, reward_n, done_n, info = env.step([4,4,4,4,4])
    while not all(done_n):
        obs_e, reward_n, done_n, info = env.step([4,4,4,4,4])	
        #obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
writeLog(os.getcwd()+'/count.txt', count)
