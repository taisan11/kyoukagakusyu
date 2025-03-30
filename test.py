import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class TextGenerationEnv:
    def __init__(self, vocabulary, max_length=10):
        self.vocabulary = vocabulary
        self.word2index = {word: idx for idx, word in enumerate(vocabulary)}  # 単語 → 数値の辞書
        self.max_length = max_length
        self.reset()
    
    def reset(self):
        self.current_text = []
        return self._get_state()
    
    def _get_state(self):
        # 単語のリストを数値ベクトルに変換（0埋めして固定長に）
        state = [self.word2index[word] for word in self.current_text]
        while len(state) < self.max_length:
            state.append(0)  # 0でパディング
        return np.array(state, dtype=np.float32)  # PyTorchが処理できる形式にする
    
    def step(self, action):
        word = self.vocabulary[action]
        self.current_text.append(word)
        
        reward = self._evaluate_sentence()
        done = len(self.current_text) >= self.max_length
        
        return self._get_state(), reward, done
    
    def _evaluate_sentence(self):
        return random.uniform(0, 1)  # 仮のスコア（後で改善）


# 単語リスト
vocab = ["こんにちは", "世界", "私は", "AI", "です", "素晴らしい", "天気", "ですね"]
env = TextGenerationEnv(vocab)

class DQNAgent:
    def __init__(self, vocab_size, state_size, hidden_size=128, lr=0.001):
        self.vocab_size = vocab_size
        self.state_size = state_size
        
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
    

    def generate_text(self, env, max_words=10):
        state = env.reset()  # 初期状態
        generated_text = []
        
        for _ in range(max_words):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.model(state_tensor)).item()  # 最も良い行動を選択
            
            word = env.vocabulary[action]
            generated_text.append(word)
            
            state, _, done = env.step(action)  # 次の状態を取得
            if done:
                break
        
        return " ".join(generated_text)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.vocab_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # NumPy 配列に変換（list of lists → 2D array）
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


def main():
    agent = DQNAgent(vocab_size=len(vocab), state_size=10)
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action([0] * 10)  # シンプルな初期状態（要改良）
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.train()
        
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
    print(agent.generate_text(env, max_words=5))  # 生成されたテキストを表示

if __name__ == "__main__":
    main()