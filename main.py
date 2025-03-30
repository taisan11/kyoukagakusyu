import numpy as np
import random

class Environment:
    def __init__(self):
        self.state = 0  # 初期状態
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        # 行動に基づいて状態を更新
        self.state += action
        reward = self.get_reward(self.state)
        if self.state >= 10:  # 例: 状態が10以上で終了
            self.done = True
        return self.state, reward, self.done

    def get_reward(self, state):
        print(f"State: {state}")  # 状態を表示
        # 得点の判断部分（ユーザーが実装）
        return state  # 例: 状態をそのまま報酬とする

class Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.q_table = np.zeros((20, 2))  # 状態数と行動数に基づくQテーブル
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def choose_action(self, state):
        if random.uniform(0, 1) < 0.1:  # ε-greedy法
            return random.choice([0, 1])  # ランダムな行動
        return np.argmax(self.q_table[state])  # Q値が最大の行動を選択

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

def main():
    env = Environment()
    agent = Agent()

    for episode in range(100):  # エピソード数
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            total_reward += reward
            state = next_state

            if done:
                break

        print(f'Episode {episode + 1}: Total Reward: {total_reward}')

if __name__ == "__main__":
    main()