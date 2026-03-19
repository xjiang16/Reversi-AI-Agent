import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Reversi:
    def __init__(self):
        self.board = np.zeros((8, 8))
        self.board[3, 4] = -1
        self.board[3, 3] = 1
        self.board[4, 3] = -1
        self.board[4, 4] = 1
        self.white_count = 2
        self.black_count = 2
        self.directions = [
            [1, 1], [1, 0], [1, -1],
            [0, 1], [0, -1],
            [-1, 1], [-1, 0], [-1, -1]
        ]
        self.turn = 1

    def reset(self):
        self.__init__()

    def is_valid_move(self, x, y, turn):
        if self.board[x, y] != 0:
            return False
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx, ny] == -turn:
                    found_opponent = True
                elif self.board[nx, ny] == turn and found_opponent:
                    return True
                else:
                    break
                nx += dx
                ny += dy
        return False

    def apply_move(self, x, y, turn):
        self.board[x, y] = turn
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            tiles_to_flip = []
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx, ny] == -turn:
                    tiles_to_flip.append((nx, ny))
                elif self.board[nx, ny] == turn:
                    for fx, fy in tiles_to_flip:
                        self.board[fx, fy] = turn
                    break
                else:
                    break
                nx += dx
                ny += dy

    def step(self, x, y, turn):
        if not self.is_valid_move(x, y, turn):
            return -1, True  # Invalid move

        self.apply_move(x, y, turn)
        self.white_count = np.sum(self.board == -1)
        self.black_count = np.sum(self.board == 1)
        done = (self.white_count + self.black_count == 64) or not self.has_valid_moves(-turn)

        reward = self.black_count - self.white_count if turn == 1 else self.white_count - self.black_count
        return reward, done

    def has_valid_moves(self, turn):
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y, turn):
                    return True
        return False

    def get_valid_actions(self, turn):
        actions = []
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y, turn):
                    actions.append((x, y))
        return actions

class DQNPlayer:
    def __init__(self):
        self.game = Reversi()
        self.state_size = 64
        self.action_size = 64
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy().flatten()
        valid_action_indices = [x * 8 + y for x, y in valid_actions]
        best_action_index = max(valid_action_indices, key=lambda a: q_values[a])
        return divmod(best_action_index, 8)

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor).clone().detach()
            action_index = action[0] * 8 + action[1]
            target_f[0][action_index] = target

            loss = nn.MSELoss()(self.model(state_tensor), target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_self_play(self, num_episodes=500, batch_size=32):
        for e in range(num_episodes):
            self.game.reset()
            state = self.game.board.flatten()
            done = False
            turn = 1

            while not done:
                valid_actions = self.game.get_valid_actions(turn)
                if not valid_actions:
                    done = True
                    break

                action = self.act(state, valid_actions)
                reward, done = self.game.step(action[0], action[1], turn)
                next_state = self.game.board.flatten()

                self.remember(state, action, reward, next_state, done)

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

                state = next_state
                turn = -turn

            print(f"Episode {e+1}/{num_episodes} finished")

def main():
    player = DQNPlayer()
    player.train_self_play()

if __name__ == "__main__":
    main()
