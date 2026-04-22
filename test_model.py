import torch
import numpy as np
import pickle
import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


map_row = 529
map_col = 564
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 4 modes
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 4)
        self.fc3 = torch.nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))  # relu activation function
        # return self.fc2(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)


class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc0 = torch.nn.Linear(state_dim, hidden_dim)  # 4 modes
        self.fcV = torch.nn.Linear(hidden_dim, 1)
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        A = F.elu(self.fc0(x))
        A = self.fcA(A)

        V = F.elu(self.fc0(x))
        V = self.fcV(V)

        Q = V + A - A.mean(1).view(-1, 1)

        return Q


class simpled_VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(simpled_VAnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc0 = torch.nn.Linear(state_dim, hidden_dim)  # 4 modes
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)
        self.fcV = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc0(x)
        A = self.fcA(F.relu(x))
        V = self.fcV(F.relu(x))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


# SAC method to balance the exploration and exploitation
class Policy(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 4)
        self.fc3 = torch.nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        # use softmax to get the probability of each action
        return F.softmax(self.fc3(x), dim=1)


class PolicyWithConv(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, conv_dim):
        super(PolicyWithConv, self).__init__()
        self.conv_net = ConvNet(input_channel=1, output_dim=conv_dim)
        self.fc1 = nn.Linear(state_dim + conv_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, map_info):
        """
        :param state: start and end position
        :param map_info: 5x5 matrix neighbor info
        :return:
        """

        conv_output = self.conv_net(map_info.unsqueeze(1))
        combined_input = torch.cat([state, conv_output], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy, gamma, tau, device, with_conv=False,
                 using_realmap=False, mapdata=None):
        # policy net
        self.actor = Policy(state_dim, hidden_dim, action_dim).to(device)
        # actor and critic net using VAnet
        self.critic1 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = VAnet(state_dim, hidden_dim, action_dim).to(device)

        # set the optimizer & initialize the target net
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # set the hyperparameters & device
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.visited_states = set()  # Set to store visited states

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action_prob = self.actor(state)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()

        # Avoid revisiting states
        # for _ in range(action_prob.shape[1]):
        #     if tuple(state[:2]) in self.visited_states:
        #         action_prob[0, action] = -float('inf')

        self.visited_states.add(tuple(state[:2]))
        return action.item()

    def calculate_target(self, rewards, next_states, done):
        next_probs = self.actor(next_states)
        next_logprobs = torch.log(next_probs + 1e-8)  # add a small value to avoid NAN
        ent = -torch.sum(next_probs * next_logprobs, dim=1).unsqueeze(1)
        q1_value = self.target_critic1(next_states)
        q2_value = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs * (torch.min(q1_value, q2_value))
                               , dim=1).unsqueeze(1)
        next_value = min_qvalue + self.log_alpha.exp() * ent
        td_target = rewards + self.gamma * (1 - done) * next_value

        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        # update the critic net
        td_target = self.calculate_target(rewards, next_states, dones)
        critic_q1_values = self.critic1(states).gather(1, actions)
        critic_q2_values = self.critic2(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.smooth_l1_loss(critic_q1_values, td_target.detach()))
        critic_2_loss = torch.mean(
            F.smooth_l1_loss(critic_q2_values, td_target.detach()))

        # optimize the critic net
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update the policy net

        probs = self.actor(states)
        logprobs = torch.log(probs + 1e-8)
        ent = -torch.sum(probs * logprobs, dim=1).unsqueeze(1)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = torch.sum(probs * (torch.min(q1_value, q2_value)), dim=1).unsqueeze(1)
        actor_loss = torch.mean(- self.log_alpha.exp() * ent - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the alpha
        alpha_loss = torch.mean((ent - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # soft update the target net
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return float(critic_1_loss), float(actor_loss)


class SACWithConv(SAC):
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy, gamma, tau, device,
                 using_realmap=False, mapdata=None, mode='GSD'):
        super(SACWithConv, self).__init__(state_dim, hidden_dim, action_dim,
                                          actor_lr, critic_lr,
                                          alpha_lr, target_entropy, gamma, tau, device,
                                          using_realmap=False, mapdata=None)
        self.mode = mode
        self.mapdata = mapdata
        self.mode_mapdata = self.mapdata[self.mode]
        self.actor = PolicyWithConv(state_dim, hidden_dim, action_dim, 5 * 5).to(device)

    def set_mode(self, mode):
        self.mode = mode
        self.mode_mapdata = self.mapdata[self.mode]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        current_positions = torch.tensor(transition_dict['current_position'], dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        # update the critic net
        td_target = self.calculate_target_with_conv(rewards, next_states, transition_dict['next_position'], dones)
        critic_q1_values = self.critic1(states).gather(1, actions)
        critic_q2_values = self.critic2(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_q1_values, td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(critic_q2_values, td_target.detach()))

        # optimize the critic net
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update the policy net
        sensation_matrix = sense_map(self.mode_mapdata, current_positions, grid=5)
        probs = self.actor(states, sensation_matrix)
        logprobs = torch.log(probs + 1e-8)
        ent = -torch.sum(probs * logprobs, dim=1).unsqueeze(1)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = torch.sum(probs * (torch.min(q1_value, q2_value)), dim=1).unsqueeze(1)
        actor_loss = torch.mean(- self.log_alpha.exp() * ent - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the alpha
        alpha_loss = torch.mean((ent - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # soft update the target net
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return float(critic_1_loss), float(actor_loss)

    def calculate_target_with_conv(self, rewards, next_states, next_positions, done):
        sensation_matrix = sense_map(self.mode_mapdata, next_positions, grid=5)
        next_probs = self.actor(next_states, sensation_matrix)
        next_logprobs = torch.log(next_probs + 1e-8)
        ent = -torch.sum(next_probs * next_logprobs, dim=1).unsqueeze(1)
        q1_value = self.target_critic1(next_states)
        q2_value = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs * (torch.min(q1_value, q2_value))
                               , dim=1).unsqueeze(1)
        next_value = min_qvalue + self.log_alpha.exp() * ent
        td_target = rewards + self.gamma * (1 - done) * next_value

        return td_target

    def take_action_with_conv(self, state, position):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        position = torch.tensor([position], dtype=torch.float).to(self.device)
        sensation_matrix = sense_map(self.mode_mapdata, position, grid=5)
        action_prob = self.actor(state, sensation_matrix)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()

        return action.item()


class SAC_2agent(SAC):
    pass

# ===============================================
# VALUE
# ===============================================

# DQN method, Q-net for DQN and VA-net for duelDQN, DQN method is unstable and the return will wander
class DQN:
    # DQN foe discrete action space
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 lr,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type,
                 using_realmap=False,
                 mode=None):
        self.using_realmap = using_realmap
        self.dqn_type = dqn_type
        self.action_dim = action_dim

        if dqn_type == "dueling":
            # set the Q-net
            self.qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)
            # set the target Q-net
            self.target_qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
            # set the Q-net
            self.qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)
            # set the target Q-net
            self.target_qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon-greedy
        self.target_update = target_update
        self.device = device
        self.cnt = 0
        self.visited_states = set()  # Set to store visited states
        self.mode = mode

    def _get_feasible_actions(self, state, delta, size='5x5') ->list[int]:
        # state:[pos_x, pos_y, end_x, end_y, delta_x, delta_y]
        realmap_x = int(state[0]+delta[0])
        realmap_y = int(state[1]+delta[1])
        action_set = set()
        actionlist = []
        if self.using_realmap: # TODO: API for mode identification, now all modes are GG when train and TS when test
            movements = dxdy_dict
            if size == '5x5':
                # Define the 3x3 sub-grids and their corresponding actions
                sub_grids = {
                    (0, 0): [0, 1, 2],  # Bottom-left
                    (0, 1): [2, 3, 4],  # Bottom-right
                    (1, 0): [4, 5, 6],  # Top-left
                    (1, 1): [6, 7, 0]  # Top-right
                }

                for (dx, dy), actions in sub_grids.items():
                    for i in range(3):
                        for j in range(3):
                            new_x, new_y = realmap_x + dx * 2 + i, realmap_y + dy * 2 + j
                            if 0 <= new_x < len(self.mapdata["GG"]) and 0 <= new_y < len(self.mapdata["GG"][0]):
                                if self.mapdata["GG"][new_x][new_y] == 1:
                                    action_set.update(actions)
                actionlist = list(action_set)

            if size=='3x3':
                for action, (dx, dy) in movements.items():
                    new_x, new_y = realmap_x + dx, realmap_y + dy
                    if 0 <= new_x < len(self.mapdata["GG"]) and 0 <= new_y < len(self.mapdata["GG"][0]):
                        if self.mapdata["GG"][new_x][new_y] == 1:
                            actionlist.append(action)

        return actionlist

    def take_action(self,state)->int: # use epsilon-greedy to take action
        # Avoid step not in feasible actions
        #feasible_action = self._get_feasible_actions(state, delta, size='3x3')
        #print(feasible_action,state)

        if np.random.random()< self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            # the q_values is self.qnet(state) for DQN and self.qnet(state).max(1)[0] for duelDQN
            q_values = self.qnet(state_tensor)

            action = q_values.argmax().item()

        #     # Avoid revisiting states
        #     for _ in range(self.action_dim):
        #         if tuple(state[:2]) in self.visited_states: #or action not in feasible_action:
        #             q_values[0, action] = -float('inf')  # Set Q-value to negative infinity and resample
        #             action = q_values.argmax().item()
        #         else:
        #             break
        # self.visited_states.add(tuple(state[:2]))
        return action

    def take_2d_action(self, state, delta) -> tuple:
        actions = [-1, 0, 1]
        if np.random.random() < self.epsilon:
            action_x = np.random.choice(actions)
            action_y = np.random.choice(actions)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            q_values = self.qnet(state_tensor)
            action_x = actions[q_values[0, 0].argmax().item()]
            action_y = actions[q_values[0, 1].argmax().item()]

        return (action_x, action_y)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'],
                                dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)

        # double DQN next 2 lines
        max_action = self.qnet(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_qnet(next_states).gather(1, max_action)

        # DQN
        #max_next_q_values = self.target_qnet(next_states).max(1)[0].view(
        #    -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # print('q_values',q_values,'q_targets:', q_targets)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        loss = float(dqn_loss)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.cnt % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict())  #
        self.cnt += 1
        return loss

class Qnet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim ,action_dim):
        super(Qnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc1 = torch.nn.Linear(state_dim, 128) # 4 modes
        self.fc2 = torch.nn.Linear(128, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(4, action_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))  # relu activation function
        # return self.fc2(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        return self.fc3(x)

class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc0 = torch.nn.Linear(state_dim, 128) # 4 modes

        self.fc1 = torch.nn.Linear(128, hidden_dim)
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)
        self.fcV = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        A = self.fcA(F.relu(self.fc1(x)))
        V = self.fcV(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q



class ConvNet(nn.Module):
    def __init__(self, input_channel, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 1, 3, 1, 1)
        self.fc1 = nn.Linear(1 * 5 * 5, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def mapdata_to_modelmatrix(mapdata: dict, n_row, n_col) -> dict[str: list[list:int]]:
    """
    Convert the mapdata to a matrix that can be used as input to the lower_model
    :param mapdata: dict, the mapdata
    :return: dict, the matrix that can be used as input to the lower_model
    """
    modelmatrix = {"TG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                   "GG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                   "GSD": [[0 for _ in range(n_row)] for _ in range(n_col)],
                   "TS": [[0 for _ in range(n_row)] for _ in range(n_col)]
                   }
    for k, v in mapdata.items():
        try:
            if v[4] & 1 == 1 or v[4] >> 1 & 1 == 1:
                modelmatrix['TG'][k[0]][k[1]] = 1
            if v[4] & 1 == 1 or v[4] >> 6 & 1 == 1 or v[4] >> 1 & 1 == 1:
                modelmatrix['TS'][k[0]][k[1]] = 1
            if v[4] >> 3 & 1 == 1:
                modelmatrix['GG'][k[0]][k[1]] = 1
            if v[4] >> 2 & 1 == 1 or v[4] >> 5 & 1 == 1:
                modelmatrix['GSD'][k[0]][k[1]] = 1
        except:

            print('Inout Data Out of Range: ', k, v, 'Map Size: ', n_row, n_col)
    return modelmatrix


def get_neighbor(modelmatrix, x, y, size=3) -> list:
    """
    Get the neighbor of the grid (x, y)
    :param modelmatrix: list[list], the modelmatrix
    :param x: int, the x coordinate of the grid
    :param y: int, the y coordinate of the grid
    :return: list, the neighbor of the grid (x, y) from (1,0) to (1,-1)
    """
    try:
        xmax = len(modelmatrix)
        ymax = len(modelmatrix[0])
    except:
        print('Input Data Out of Range When Getting Neighbor: ', type(modelmatrix), x, y)
        return [0 for _ in range(size * size)]

    neighbors = []
    if size == 3:
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for dx, dy in directions:
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                neighbors.append(modelmatrix[nx][ny])
            else:
                neighbors.append(0)  # or some other value indicating out of bounds

    if size == 5:
        directions = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

        for dx, dy in directions:
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                neighbors.append(modelmatrix[nx][ny])
            else:
                neighbors.append(0)

    return neighbors


def sense_map(mapdata, position_tensor, grid=5):
    """
    Sense the mapdata at the grid (x, y)
    :param mapdata: list[list], the mapdata
    :param position_tensor: torch.tensor, the position tensor, [[x1, y1], [x2, y2], ...[xn, yn]] x m
    :param grid: int, the size of the grid (default is 5)
    :return: tensor size (n, grid, grid)
    """

    def _inner(mapdata, x, y):
        xmax = len(mapdata)
        ymax = len(mapdata[0])
        sensed_data = [[0 for _ in range(grid)] for _ in range(grid)]
        half_grid = grid // 2

        for i in range(grid):
            for j in range(grid):
                nx, ny = x - half_grid + i, y - half_grid + j
                if 0 <= nx < xmax and 0 <= ny < ymax:
                    sensed_data[i][j] = mapdata[nx][ny]
                else:
                    sensed_data[i][j] = 0  # or some other value indicating out of bounds
        sensed_data = torch.tensor(sensed_data).unsqueeze(0).float().to(torch.device("cpu"))
        return sensed_data

    position_tensor = list(position_tensor)
    return torch.cat([_inner(mapdata, int(x), int(y)) for x, y in position_tensor], dim=0)


# ===============================================
# ENV
# ===============================================

# glabal variables
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
modelist = ['GSD', 'GG', 'TS', 'TG']
# with open('/home/g/software/pycharm/projects/RLTRAJ-origin/data/vdistributiondatainrealworld.pkl', 'rb') as f:
#     processed_data = pickle.load(f)
with open('vdistributiondatainrealworld.pkl', 'rb') as f:
    processed_data = pickle.load(f)


# print(processed_data)

def _allow(neighbor: int, mode: str) -> bool:
    """
    check if the neighbor is allowed to travel of the given mode: static, TG, GG, GSD, TS
    :param neighbor: int, element of a list of 9 elements, the 4-th element is the grid itself, the other 8 elements are the neighbors
    :param mode: str, 'TG', 'GG', 'GSD', 'static'
    :return: bool, True if the neighbor is allowed to travel of the given mode, False otherwise
    """
    if mode == 'TG' or mode == 'static':
        return neighbor >> 1 & 1 == 1
    elif mode == 'GG':
        return neighbor >> 3 & 1 == 1
    elif mode == 'GSD':
        return neighbor >> 2 & 1 == 1 or neighbor >> 5 & 1 == 1
    elif mode == 'TS':
        return neighbor >> 6 & 1 == 1 or neighbor >> 1 & 1 == 1
    elif mode == 'static':
        return True


class UpperEnv:
    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 trainid_start=0, test_mode=False, testid_start=0,
                 test_num=8, train_num=10000,
                 m=5,
                 use_real_map=False, realmap_row=326, realmap_col=364, lower_model_config=None):
        """

        :param mapdata: dict of 5 elements, key is the mode, value is the mapdata of the mode
        :param traj: traj data in csv format
        :param trainid_start: start index of the training data
        :param test_mode: ==True if test
        :param testid_start:  start index of the test data
        :param test_num: pass
        :param train_num: pass
        :param use_real_map: default is False
        :param realmap_row: pass
        :param realmap_col: pass
        :param m: the hyperpara of steps to calculate the avg reward
        :param lower_model_config: dict of lower model(qnet) size key=state_dim, hidden_dim, action_dim,model_path
        """
        self.step_cnt = 0
        self.traj = traj
        self.traj_idx = 0
        self.train_num = train_num
        self.trainid_start = trainid_start
        self.m = m
        self.max_step = 20  # max step of the upper model
        self.realmap_row = realmap_row
        self.realmap_col = realmap_col

        if use_real_map:
            print('using real map', realmap_row, realmap_col)
            self.mapdata = mapdata
            self.mapmatrice = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num

        if self.isTest:
            self.mod = self.test_num
            start_id = self.testid_start
        else:
            self.mod = self.train_num
            start_id = self.trainid_start

        # import lower model
        try:
            self.lower_type = lower_model_config['model_type']  # str , 'DQN' or 'SAC' or other
            state_dim_lower = lower_model_config['state_dim']
            hidden_dim_lower = lower_model_config['hidden_dim']
            action_dim_lower = lower_model_config['action_dim']
            model_path = lower_model_config['model_path']
        except:
            raise ValueError('lower_model_config is not correct, check the key of the dict or model path')

        lower_agent = None
        if self.lower_type == 'DQN':
            lower_agent = VAnet(state_dim_lower, hidden_dim_lower, action_dim_lower)
        elif self.lower_type == 'SAC':
            lower_agent = Policy(state_dim_lower, hidden_dim_lower, action_dim_lower)
        lower_agent.load_state_dict(torch.load(model_path))
        lower_agent.eval()
        self.lower_agent = lower_agent

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            # if self.mapmatrice[upper_mode][x][y] == 0:
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
            #     # print('lower path is blocked, and travel in rural area', x, y)
            # else:
            #     self.is_match_compute_tuple[1] += 1
            #     # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
            #     #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward = -float(abs(t_lower - t_upper)) / max(t_lower,
        #                                               t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        reward = (match_rate - 1) * float(abs(t_lower - t_upper))
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        cos = 1
        if self.step_cnt > 1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 2, ['locx_o',
                                                                                           'locy_o']])  # TODO 凡是涉及到locx locy都需要修改
            inner_product = (
                    (goal_pos[0] - start_pos[0]) * (goal_pos[0] - pre_pos[0]) + (goal_pos[1] - start_pos[1]) * (
                    goal_pos[1] - pre_pos[1]))
            length_product2 = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) * (
                    (goal_pos[0] - pre_pos[0]) ** 2 + (goal_pos[1] - pre_pos[1]) ** 2)
            cos = inner_product / (length_product2 ** 0.5 + 1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        # relative_dis = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) ** 0.5
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums), reward, done

    def step_with20action(self, action: int):
        """

        :param action:int , [0,20,30...400] denoted discreted v
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TG', 'TS']
        print(processed_data)
        matchedmode = None
        min_v_bias = 400
        for mode in action_mode_duels:
            if min_v_bias > abs(action * 17 - processed_data[mode]['mean']):
                matchedmode = mode
            min_v_bias = min(abs(action * 17 - processed_data[mode]['mean']), min_v_bias)

        upper_mode = matchedmode
        reward = 0

        # t_lower is the time cost of the lower model to reach the goal
        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=False, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = []
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
            sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
            for j in range(len(sorted_actions)):
                tmp_action = sorted_actions[j]
                if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                    lower_action = tmp_action
                    break
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)

        t_lower = 0
        v_expected = processed_data[upper_mode]['mean'] / 60
        v_rural = 0.3
        v_lower = 0

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            v_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(lower_path[i][1] - lower_path[i - 1][1]))
            # restrict x,y to avoid OutOfRange err
            if (0 <= x < self.realmap_col and 0 <= y < self.realmap_row) and self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                    lower_path[i][1] - lower_path[i - 1][1])) / v_rural
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                    lower_path[i][1] - lower_path[i - 1][1])) / v_expected

        # t_upper calculated by the given data
        idx = (self.traj_idx + self.step_cnt) % self.mod
        if idx == 0:
            t_upper = 0
        else:
            t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod + 1, 'time'] \
                          - self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod, 'time'])

        if self.isTest:
            print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
                  upper_mode)

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward -= abs(t_lower - t_upper) *(1 - (self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model
        self.step_cnt += 1
        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])

        start_idx = (self.traj_idx + self.step_cnt) % self.mod - 1
        # TODO: check the start_idx WARNING
        if start_idx < 0:
            start_idx += 1
        start_pos = tuple(self.traj.loc[start_idx, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(start_idx + 1) % self.mod, ['locx_o', 'locy_o']])
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = (abs(goal_pos[0] - start_pos[0]) ** 2 + abs(goal_pos[1] - start_pos[1]) ** 2) ** 0.5 / (
                t_upper + 1) * 60
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        # calculate reward using v
        v_lower = v_lower / (t_lower + 1) * 60
        # reward -= (abs(v_lower-v_expected)) *(1-(self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        mean, std = processed_data[upper_mode]['mean'], processed_data[upper_mode]['std']
        z_score = (v_upper - mean) / std
        # factor_dict = {'GSD':1.5, 'GG':1, 'TS':0.75, 'TG':0.75}
        reward -= abs(z_score) * (1 - self.is_match_compute_tuple[1] / (
                self.is_match_compute_tuple[0] + 0.1))  # *factor_dict[upper_mode]

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False
        v_differ = [v_upper - processed_data[i]['mean'] for i in modelist]
        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = (abs((goal_pos[0] - start_pos[0]) ** 2 + abs(goal_pos[1] - start_pos[1])) ** 2) ** 0.5
        self.upper_mode = upper_mode
        return np.array([v_upper] + v_differ + self.rts_nums), reward, done

    def reset(self):
        """
        state of upper model in time t is s_t = [r_avg, r_t-1, a_t-1] avg means the avg reward of the last m steps
        :return:self.state: np.array, the state of the environment
        """

        self.r_avg = 0
        self.step_cnt = 0
        self.max_step = 1

        self.traj_idx += 1
        i = self.traj_idx
        # print('reset upper',self.traj.loc[i,'ID'], self.traj.loc[i+1, 'ID'])
        while self.traj.loc[i % self.mod, 'ID'] == self.traj.loc[(i + 1) % self.mod, 'ID']:
            self.max_step += 1
            i += 1
        # print('upper reset', 'maxstep',self.max_step,'trajstart', self.traj_idx)

        # embedding the map info to the state
        start_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor
        self.v_avg = 0

        # relative_pos = [goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]]
        # relative_dis = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) ** 0.5
        # relative_dis = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums)


class UpperEnvRewardMathrate(UpperEnv):
    "丁老师说的匹配度作为奖励"

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            # if self.mapmatrice[upper_mode][x][y] == 0:
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
            #     # print('lower path is blocked, and travel in rural area', x, y)
            # else:
            #     self.is_match_compute_tuple[1] += 1
            #     # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
            #     #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward = -float(abs(t_lower - t_upper)) / max(t_lower,
        #                                               t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        # reward = -float(abs(t_lower - t_upper))
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        reward = match_rate
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor
        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        if self.step_cnt >= self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.0001) * 60] + self.rts_nums), reward, done

    def reset(self):
        """
        state of upper model in time t is s_t = [r_avg, r_t-1, a_t-1] avg means the avg reward of the last m steps
        :return:self.state: np.array, the state of the environment
        """

        self.r_avg = 0
        self.step_cnt = 0
        self.max_step = 1

        self.traj_idx += 1
        i = self.traj_idx
        # print('reset upper',self.traj.loc[i,'ID'], self.traj.loc[i+1, 'ID'])
        while self.traj.loc[i % self.mod, 'ID'] == self.traj.loc[(i + 1) % self.mod, 'ID']:
            self.max_step += 1
            i += 1
        # print('upper reset', 'maxstep',self.max_step,'trajstart', self.traj_idx)

        # embedding the map info to the state
        start_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor
        self.v_avg = 0
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.0001) * 60] + self.rts_nums)


class UpperEnvRewardVelocity(UpperEnv):

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            # if self.mapmatrice[upper_mode][x][y] == 0:
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
            #     # print('lower path is blocked, and travel in rural area', x, y)
            # else:
            #     self.is_match_compute_tuple[1] += 1
            #     # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
            #     #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward = -float(abs(t_lower - t_upper)) / max(t_lower,
        #                                               t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        # reward = -float(abs(t_lower - t_upper))
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        v_true = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance'] / (
                self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'] + 0.001)
        reward = match_rate / abs(v_expected - v_true)
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        cos = 1
        if self.step_cnt > 1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 2, ['locx_o',
                                                                                           'locy_o']])  # TODO 凡是涉及到locx locy都需要修改
            inner_product = (
                    (goal_pos[0] - start_pos[0]) * (goal_pos[0] - pre_pos[0]) + (goal_pos[1] - start_pos[1]) * (
                    goal_pos[1] - pre_pos[1]))
            length_product2 = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) * (
                    (goal_pos[0] - pre_pos[0]) ** 2 + (goal_pos[1] - pre_pos[1]) ** 2)
            cos = inner_product / (length_product2 ** 0.5 + 1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums), reward, done

    def reset(self):
        """
        state of upper model in time t is s_t = [r_avg, r_t-1, a_t-1] avg means the avg reward of the last m steps
        :return:self.state: np.array, the state of the environment
        """

        self.r_avg = 0
        self.step_cnt = 0
        self.max_step = 1

        self.traj_idx += 1
        i = self.traj_idx
        # print('reset upper',self.traj.loc[i,'ID'], self.traj.loc[i+1, 'ID'])
        while self.traj.loc[i % self.mod, 'ID'] == self.traj.loc[(i + 1) % self.mod, 'ID']:
            self.max_step += 1
            i += 1
        # print('upper reset', 'maxstep',self.max_step,'trajstart', self.traj_idx)

        # embedding the map info to the state
        start_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor
        self.v_avg = 0

        # relative_pos = [goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]]
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        # relative_dis = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums)


class UpperEnvRewardVelocityZscore(UpperEnvRewardVelocity):

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            # if self.mapmatrice[upper_mode][x][y] == 0:
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
            #     # print('lower path is blocked, and travel in rural area', x, y)
            # else:
            #     self.is_match_compute_tuple[1] += 1
            #     # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
            #     #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward = -float(abs(t_lower - t_upper)) / max(t_lower,
        #                                               t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        # reward = -float(abs(t_lower - t_upper))
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        v_true = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance'] / (
                self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'] + 0.001)
        reward = match_rate / abs(v_true * 60 - processed_data[upper_mode]["mean"]) * processed_data[upper_mode]["std"]
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        cos = 1
        if self.step_cnt > 1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 2, ['locx_o',
                                                                                           'locy_o']])  # TODO 凡是涉及到locx locy都需要修改
            inner_product = (
                    (goal_pos[0] - start_pos[0]) * (goal_pos[0] - pre_pos[0]) + (goal_pos[1] - start_pos[1]) * (
                    goal_pos[1] - pre_pos[1]))
            length_product2 = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) * (
                    (goal_pos[0] - pre_pos[0]) ** 2 + (goal_pos[1] - pre_pos[1]) ** 2)
            cos = inner_product / (length_product2 ** 0.5 + 1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums), reward, done


class UpperEnvRewardVelocityNoramalDistribution(UpperEnvRewardVelocityZscore):

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.realmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            if not 0<=(lower_state[:2] + lower_env.delta)[0]< self.realmap_col or not 0<=(lower_state[:2] + lower_env.delta)[1]<self.realmap_row:
                break
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            # if self.mapmatrice[upper_mode][x][y] == 0:
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
            #     # print('lower path is blocked, and travel in rural area', x, y)
            # else:
            #     self.is_match_compute_tuple[1] += 1
            #     # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
            #     #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
            #     t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
            #             lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_expected
        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward = -float(abs(t_lower - t_upper)) / max(t_lower,
        #                                               t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        # reward = -float(abs(t_lower - t_upper))
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        v_true = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance'] / (
                self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'] + 0.001)
        pdfs = {'GSD': 0, 'GG': 0, 'TS': 0, 'TG': 0}
        from scipy.stats import norm
        for mode in modelist:
            pdfs[mode] = norm.pdf(v_true * 60, processed_data[mode]["mean"], processed_data[mode]["std"])
        # Normalize the values in the `pdfs` dictionary
        # total = sum(pdfs.values())
        # if total > 0:  # Avoid division by zero
        #     pdfs = {mode: value / total for mode, value in pdfs.items()}
        reward = match_rate*pdfs[upper_mode]
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        cos = 1
        if self.step_cnt > 1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 2, ['locx_o',
                                                                                           'locy_o']])  # TODO 凡是涉及到locx locy都需要修改
            inner_product = (
                    (goal_pos[0] - start_pos[0]) * (goal_pos[0] - pre_pos[0]) + (goal_pos[1] - start_pos[1]) * (
                    goal_pos[1] - pre_pos[1]))
            length_product2 = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) * (
                    (goal_pos[0] - pre_pos[0]) ** 2 + (goal_pos[1] - pre_pos[1]) ** 2)
            cos = inner_product / (length_product2 ** 0.5 + 1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'distance']
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums), reward, done


class MapEnv:
    """
    """

    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 test_mode=False, testid_start=0, test_num=8,
                 use_real_map=False, realmap_row=326, realmap_col=364,
                 is_lower=False, dummy_mode=None):
        '''
        这里的traj用的是重新排版后的，每一条数据包含了起点和终点，这样就不需要写很多特判逻辑
        '''

        self.traj = traj
        self.step_cnt = 0
        self.map_row = realmap_row
        self.map_col = realmap_col

        if use_real_map:
            self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.is_lower = is_lower  # when is_lower is True, the env is used for lower model, and mode info is not used
        self.dummy_mode = dummy_mode

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num
        self.distance_hold = 1.5 if test_mode else 0
        self.traj_cnt = 0  # traj_CNT 是当前训练轨迹的严格索引

    def reset(self):
        # reset env by using next two traj record
        self.step_cnt = 0

        if self.isTest:
            mod = self.test_num
            start_id = self.testid_start
        else:
            mod = len(self.traj)
            start_id = 0

        # 循环直到拿到一条没有 NaN 坐标的轨迹
        while True:
            idx = start_id + self.traj_cnt % mod
            locx_start = self.traj.loc[idx, 'loc_x']
            locy_start = self.traj.loc[idx, 'loc_y']
            locx_end   = self.traj.loc[idx, 'next_loc_x']
            locy_end   = self.traj.loc[idx, 'next_loc_y']

            if any(np.isnan(v) for v in [locx_start, locy_start, locx_end, locy_end]):
                print(f"skip traj {idx} because of NaN:",
                      f"locx_start={locx_start}, locy_start={locy_start},",
                      f"locx_end={locx_end}, locy_end={locy_end}")
                self.traj_cnt += 1
                continue
            else:
                locx_start = float(locx_start)
                locy_start = float(locy_start)
                locx_end   = float(locx_end)
                locy_end   = float(locy_end)
                break

        self.mode = self.traj.loc[idx, 'mode'] if not self.is_lower else self.dummy_mode

        # 这里先用 TG 图层，如果之后想按 mode 选图层再改
        self.neighbor = np.array(get_neighbor(self.mapdata['TG'], locx_start, locy_start, size=3))
        self.delta = np.array([locx_start, locy_start])
        self.state = np.array([0, 0])
        self.goal = np.array([locx_end - locx_start, locy_end - locy_start])
        # max step is the mahattan distance between start and goal add 10
        self.max_step = np.abs(locx_start - locx_end) + np.abs(locy_start - locy_end) + 10

        self.traj_cnt += 1
        return np.hstack((self.state, self.goal, self.neighbor))

    def step(self, action: int):  # todo 这里要改
        # agent will move to 8 directions,action is tuple of (dx,dy)
        reward = 0
        self.step_cnt += 1
        d = dxdy_dict[action]

        # update state of position
        self.state += np.array(d)
        # Python
        dist = float(np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]))
        denominator = np.abs(self.goal[0]) + np.abs(self.goal[1])

        # if denominator != 0:
        #     dist /= denominator
        # else:
        #     print("Warning: denominator is zero, setting distance to infinity")
        #     dist = float('inf')  # Assign a large value or handle appropriately
        # not in the available neighbor
        if self.neighbor[action] != 0:  # when size = 3
            # if self.neighbor[(2+dx)*3 + (2+dy)]!=0: # when size=5
            #     dx,dy = d[0], d[1]
            reward += 0.65

        # update neighbor
        self.neighbor = np.array(
            get_neighbor(self.mapdata['TG'], self.state[0] + self.delta[0], self.state[1] + self.delta[1], size=3))

        # to encourage the agent travel in the shortest path
        reward -= 1 if dist > self.distance_hold else 0  # 惩罚到终点的距离
        # reward -= 1 if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > self.distance_hold else 0
        # if np.abs(self.state[0] - self.goal[0]) + np.abs(
        #         self.state[1] - self.goal[1]) == self.distance_hold or self.step_cnt == self.max_step:
        #     done = True
        # else:
        #     done = False
        if dist <= self.distance_hold:
            done = True
            reward += 2  # 找到终点给大奖励
        elif self.step_cnt >= self.max_step:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return np.hstack((self.state, self.goal, self.neighbor)), reward, done


class MapEnvRewardFunctionOne(MapEnv):
    """奖励函数化尝试1：每一步惩罚未完成的比例，走在道路上则惩罚降低"""
    def step(self, action: int):  # todo 这里要改
        # agent will move to 8 directions,action is tuple of (dx,dy)
        reward = 0
        self.step_cnt += 1
        d = dxdy_dict[action]

        # update state of position
        self.state += np.array(d)
        # Python
        dist = float(np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]))
        dist_origin = np.abs(self.goal[0]) + np.abs(self.goal[1])

        if dist_origin == 0:
            dist_origin = 1
        # not in the available neighbor
        if self.neighbor[action] != 0:  # when size = 3
            # if self.neighbor[(2+dx)*3 + (2+dy)]!=0: # when size=5
            #     dx,dy = d[0], d[1]
            reward += 0.65 * dist / dist_origin  # 0.65 is the reward for moving to a neighbor cell, normalized by the distance to goal

        # update neighbor
        self.neighbor = np.array(
            get_neighbor(self.mapdata[self.mode], self.state[0] + self.delta[0], self.state[1] + self.delta[1], size=3))

        # to encourage the agent travel in the shortest path
        reward -= dist / dist_origin if dist > self.distance_hold else 0  # 惩罚到终点的距离
        # reward -= 1 if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > self.distance_hold else 0
        # if np.abs(self.state[0] - self.goal[0]) + np.abs(
        #         self.state[1] - self.goal[1]) <= self.distance_hold or self.step_cnt >= self.max_step:
        #     done = True
        # else:
        #     done = False
        if dist <= self.distance_hold:
            done = True
            reward += 2  # 找到终点给大奖励
        elif self.step_cnt >= self.max_step:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return np.hstack((self.state, self.goal, self.neighbor)), reward, done


class ODMapEnv:
    """
    训练两个网络两个智能体，一个从起点出发，一个从终点出发，till met or max step
    """

    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 test_mode=False, testid_start=0, test_num=8,
                 use_real_map=False, realmap_row=326, realmap_col=364,
                 is_lower=False, dummy_mode=None):
        '''
        这里的traj用的是重新排版后的，每一条数据包含了起点和终点，这样就不需要写很多特判逻辑
        '''

        self.traj = traj
        self.step_cnt = 0
        self.map_row = realmap_row
        self.map_col = realmap_col

        if use_real_map:
            self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.is_lower = is_lower  # when is_lower is True, the env is used for lower model, and mode info is not used
        self.dummy_mode = dummy_mode

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num
        self.distance_hold = 0 if test_mode else 1
        self.traj_cnt = 0  # traj_CNT 是当前训练轨迹的严格索引

    def reset(self):
        """
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        # 考虑到两个智能体，所以返回state是一个dict，d[start]和d[end]分别为原先结构

        :return:  包含从起点和终点出发的智能体的state的字典
        """

        self.step_cnt = 0

        if self.isTest:
            mod = self.test_num
            start_id = self.testid_start
        else:
            mod = len(self.traj)
            start_id = 0

        # 循环直到拿到一条没有 NaN 坐标的轨迹
        while True:
            idx = start_id + self.traj_cnt % mod
            locx_start = self.traj.loc[idx, 'loc_x']
            locy_start = self.traj.loc[idx, 'loc_y']
            locx_end   = self.traj.loc[idx, 'next_loc_x']
            locy_end   = self.traj.loc[idx, 'next_loc_y']

            if any(np.isnan(v) for v in [locx_start, locy_start, locx_end, locy_end]):
                print(f"skip traj {idx} because of NaN:",
                      f"locx_start={locx_start}, locy_start={locy_start},",
                      f"locx_end={locx_end}, locy_end={locy_end}")
                self.traj_cnt += 1
                continue
            else:
                locx_start = float(locx_start)
                locy_start = float(locy_start)
                locx_end   = float(locx_end)
                locy_end   = float(locy_end)
                break

        self.mode = self.traj.loc[self.traj_cnt % mod, 'mode'] if not self.is_lower else self.dummy_mode

        # delta is the relative position of the start_position and 0,0; delta only change when start_position change(when reset)
        # neighbor is the 8 elements list of the grid not including itself, 0-8 are the neighbors from 1,0 to 1,-1
        self.neighbor_start = np.array(get_neighbor(self.mapdata[self.mode], self.locx_start, self.locy_start, size=3))
        self.neighbor_end = np.array(get_neighbor(self.mapdata[self.mode], self.locx_end, self.locy_end, size=3))

        # 计算t==0时刻相对坐标
        ego_pos = np.array([0, 0])
        goal_pos_start = np.array([self.locx_end - self.locx_start, self.locy_end - self.locy_start])
        goal_pos_end = np.array([self.locx_start - self.locx_end, self.locy_start - self.locy_end])

        # max step is the mahattan distance between start and goal add 10
        self.max_step = np.abs(self.locx_start - self.locx_end) + np.abs(self.locy_start - self.locy_end) + 10

        self.state_dual = {
            'start': np.hstack((ego_pos, goal_pos_start, self.neighbor_start)),
            'end': np.hstack((ego_pos, goal_pos_end, self.neighbor_end))
        }

        return self.state_dual

    def step_2agent(self, action_start: int, action_end: int):  #
        """
        接受两个智能体动作后更新环境，如果要多agents，传参actionlist就行

        :param action_start: 从起点出发的agent的动作
        :param action_end: 从终点出发的agent的动作
        :return: state_dual, reward_dual, done # state reward 是各自的，done是共有的
        """
        # agent will move to 8 directions,action is tuple of (dx,dy)
        self.step_cnt += 1

        r_start, r_end = 0, 0
        d_start, d_end = dxdy_dict[action_start], dxdy_dict[action_end]

        # 更新 t==t 时刻绝对坐标
        self.locx_start += d_start[0]
        self.locy_start += d_start[1]
        self.locx_end += d_start[0]
        self.locy_end += d_start[1]

        # 更新state
        for k in self.state_dual.keys():
            if k == 'start':
                self.state_dual[k][:2] += np.array([d_start[0], d_start[1]])
                self.state_dual[k][2:4] += np.array([d_end[0], d_end[1]])
                self.state_dual[k][4:] = np.array(get_neighbor(self.mapdata[self.mode],
                                                               self.locx_start,
                                                               self.locy_start, size=3))
            elif k == 'end':
                self.state_dual[k][:2] += np.array([d_end[0], d_end[1]])
                self.state_dual[k][2:4] += np.array([d_start[0], d_start[1]])
                self.state_dual[k][4:] = np.array(get_neighbor(self.mapdata[self.mode],
                                                               self.locx_end,
                                                               self.locy_end, size=3))

        # not in the available neighbor
        if self.neighbor_start[action_start] != 0:  # when size = 3
            r_start += 0.0
        if self.neighbor_end[action_end] != 0:
            r_end += 0.0

        # update neighbor
        self.neighbor_start = self.state_dual['start'][4:]
        self.neighbor_end = self.state_dual['end'][4:]

        # to encourage the agent travel in the shortest path
        if np.abs(self.locx_start - self.locx_end) + np.abs(self.locy_start - self.locy_end) > self.distance_hold:
            r_start -= 1
            r_end -= 1

        if np.abs(self.locx_start - self.locx_end) + np.abs(
                self.locy_start - self.locy_end) <= self.distance_hold or self.step_cnt == self.max_step:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return self.state_dual, r_start, r_end, done





















map_row = 529
map_col = 564
state_dim = 12
hidden_dim = 64
action_dim = 8

actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('SAC_15000_eps_inrealmap_7121829.pth', map_location=torch.device('cpu')))
actor_net.eval()

spark = (
    SparkSession.builder
    .appName("LowerPolicyEval")
    .enableHiveSupport()
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
spark.sql("USE ss_seu_df")

with open('GridModesAdjacentRealworld.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = spark.table("preprocessed_v5").toPandas()
# spark.stop()

# traj = pd.read_csv('testdata_20.csv')

env = MapEnv(mapdata, traj, test_mode=True, testid_start=0, test_num=len(traj),
             use_real_map=True, realmap_row=map_row, realmap_col=map_col)

all_trajs = []
results = []
match_rates = []
modes = [] # store the mode of each traj
num_finish = 0

for i in range(12000):
    state = env.reset()
    done = False
    total_reward = 0
    distance_to_start = state[:2] # always (0, 0)
    end_to_start = state[2:4]
    path = []
    actual_pos = []

    step_cnt = 0
    actions = []
    state_set = set()
    state_set.add(tuple(state[:2]))
    while not done:
        step_cnt += 1

        #选取动作避免重复
        action_scores = actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        sorted_actions = torch.sort(action_scores, descending=True).indices.squeeze().tolist()
        for action in sorted_actions:
            if len(actions) == 0:
                action = sorted_actions[0]
            elif not (dxdy_dict[action][0] + dxdy_dict[actions[-1]][0] == 0 and dxdy_dict[action][1] +
                      dxdy_dict[actions[-1]][1] == 0):
                break
            else:
                action = sorted_actions[0]  # Fallback to the best action if no valid action is found
        # action = int(actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())

        next_state, reward, done = env.step(action)
        actions.append(action)
        state = next_state
        state_set.add(tuple(state[:2]))
        total_reward += reward
        path.append(state[:2])
        actual_pos.append(state[:2]+ env.delta)
    all_trajs.append(actual_pos)

    total_step = 0
    match_step = 0
    # caculate match rate
    tg_map = env.mapdata['TG']
    max_x = len(tg_map)
    max_y = len(tg_map[0])

    for j, coord in enumerate(actual_pos[1:]):
        x, y = int(coord[0]), int(coord[1])

        # 越界直接跳过，避免 IndexError
        if not (0 <= x < max_x and 0 <= y < max_y):
            # print(f"out of range: x={x}, y={y}")
            continue

        if tg_map[x][y] == 1:
            match_step += 1
        total_step += 1


    mahattan_dis = abs(end_to_start[0]) + abs(end_to_start[1])
    if step_cnt <= mahattan_dis:
        match_rate = match_step / (total_step +0.0001)
        match_rates.append(match_rate)
        modes.append(env.mode)
        num_finish += 1
    else:
        print('not finished traj id:',i)
    results.append((distance_to_start, end_to_start, total_reward, path, step_cnt, actions))
    # Plot the path and the start and end points
    path = np.array(path)

print('finish rate:',num_finish/len(results), 'TOTAL:',len(results),'FINISH:',num_finish)
print('average match rate:',np.mean(match_rates))
spark.stop()
