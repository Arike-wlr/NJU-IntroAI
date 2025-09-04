import copy
from env import BaitEnv
import numpy as np
class DFSAgent:
    def __init__(self, env:BaitEnv, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed
        self.closed=[]
        self.action=[]

    def limiteddfs(self,env,actions):

        if self.tick > self.tick_max:
            assert 0

        for action_id in actions:
            env_copy = copy.deepcopy(env)
            #TODO:实现位置查找和距离计算
            next_state, reward, isOver, info = env_copy.step(action_id)
            sorted_actions=self.calculate_distance(next_state)

        for action_id in sorted_actions:
            if next_state in self.closed:  # 这边走过了或者是撞墙上了或者是推不动，应该是吧。
                continue  # 直接进入下一个方向
            self.tick += 1  # 没有走过，就步数加一
            print(f"Used steps: {self.tick} / {self.tick_max}")
            if isOver and info['message'] != 'Fell into hole. Game over.':  # 游戏结束并且没掉洞里（成功）
                self.action.append(action_id)  # 把这步加入目标
                return True
            else:  # 掉洞里或者撞墙或者就是普通的没有成功在路上//或者是没钥匙单撞门了
                if isOver:  # 掉洞里了
                    return False  # 返回到上一个函数，然后就是删掉这一步。
                elif info != {} and info['message'] == "Need key to open goal":  # 撞墙不会产生新状态，但没钥匙撞门会。
                    self.closed.append(next_state)  # 加入新状态
                    continue  # 处理和走入旧状态是一样的
                else:  # 在路上
                    self.closed.append(next_state)  # 这个新状态加进去
                    self.action.append(action_id)  # 这一步加进去
                    if self.dfs(env_copy, actions):
                        return True
                    self.action.pop()
                    self.tick -= 1  # 如果接下来的都返回False，说明这一步走不通，删掉。
        return False

    def solve(self):
        self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        actions.pop(0)
        if self.limiteddfs(self.env,actions):
            action_sequence = self.action
        return action_sequence

    def calculate_distance(self,state):
        K_indice=self.find_value(state,'key')
        if K_indice !=[]:
            A_index=self.find_value(state,"avatar_nokey")[0]
            K_index=K_indice[0]
            print(A_index,K_index)
            sorted_actions=[]
        return sorted_actions

    def find_value(self,arr, target):
        return [(i, j) for i, row in enumerate(arr)
                for j, cell in enumerate(row)
                if target in cell]

if __name__=="__main__":
    env = BaitEnv(level=0, render=True)
    env.reset()
    agent = DFSAgent(env, 30)

    initial, reward, isOver, info =env.step(0)
    print(initial)
    l=agent.calculate_distance(initial)
    #env.reset()
    #initial, reward, isOver, info = env.step(1)
    #print(initial)