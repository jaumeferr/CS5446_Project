import gym
import ma_gym
import random

rewards = 0
env = gym.make("Combat-v0")
env.reset()
ns, r, d, i = env.step([random.randint(4,4) for _ in range(5)])
while not all(d): 
    ns, r, d, i = env.step([random.randint(4,4) for _ in range(5)])
    rewards+= sum(r)
    print(r)

print(rewards)
print(env.agent_health)
print(env.opp_health)
