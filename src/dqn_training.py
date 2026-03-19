import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from reversi import reversi

class ReversiDQN(nn.Module):
    def __init__(self, board_size):
        super(ReversiDQN, self).__init__()
        self.fc1 = nn.Linear(board_size * board_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, board_size * board_size)

    def forward(self, x):
        x = x.view(-1, 64)  # Flatten the input to a 1D vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

board_size = 8
policy_net = ReversiDQN(board_size)
target_net = ReversiDQN(board_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

steps_done = 0

def select_action(state):
    global steps_done
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(board_size * board_size)]], dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_ai(num_episodes=500):
    game = reversi()
    for i_episode in range(num_episodes):
        state = game.board.flatten()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in range(100):  # Limit the number of steps per episode
            action = select_action(state)
            x, y = divmod(action.item(), board_size)

            # Check for valid move and perform it
            reward = game.step(x, y, game.turn)

            # Ensure reward is a tensor
            reward = torch.tensor([reward], dtype=torch.float32)

            # If move is invalid, penalize and continue
            if reward.item() < 0:
                next_state = state
            else:
                next_state = game.board.flatten()
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                game.turn = -game.turn

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            if reward.item() < 0:  # End the episode if an invalid move is made
                break

if __name__ == '__main__':
    train_ai()
