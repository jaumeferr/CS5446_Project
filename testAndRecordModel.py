import os
import torch.nn as nn


script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

def writeLog(filepath, count):
    file=open(filepath,"w+")
    file.write(str(count))
    file.close()

def readLog(filepath):
    count=int([line for line in open(filepath, "r")][0])
    return count

def get_model():
    model_state_dict, hx_size, recurrent = torch.load(model_path)
    model = QNet(env.observation_space, env.action_space, recurrent)
    model.load_state_dict(model_state_dict)
    return model


import gym
import ma_gym
import torch
from ma_gym.wrappers import Monitor

count = readLog(os.getcwd()+'\\count.txt')
print(type(count),":",count)
env = gym.make('Combat-v0')
env.reset()

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space) #Each Agent has its observation space. Class Box
        self.recurrent = recurrent
        self.hx_size = 32 #Originally 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            ## Possible to re-do architecture, or tune the NN hyperparameters.
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(
                                                                            nn.Linear(n_obs, 256),
                                                                            nn.ReLU(),
                                                                            nn.Linear(256, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0],)] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden) #Using the QNet to sample an action, we feed obs and get action.
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden

    def get_action(self, obs, hidden):
        action, hidden = self.forward(obs, hidden)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


target = int(input("How many simulations would you like to perform?"))

model = get_model()

for i in range(target):
    count += 1
    score = 0
    print(count)
    directory = f'ma-gym\\recordings\\{count}'
    env=Monitor(env, directory=directory)
    state=env.reset()

    with torch.no_grad():
        hidden = model.init_hidden()
        action, hidden = model.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
        print(action)
        next_state, r, d, i = env.step(action[0].data.numpy().tolist())
        state = next_state
        score += sum(r)
        while not all(d):
            action, hidden = model.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
            print(action)
            next_state, r, d, i = env.step(action[0].data.numpy().tolist())
            state = next_state
            score += sum(r)
    print(score)
writeLog(os.getcwd()+'\\count.txt',count)

