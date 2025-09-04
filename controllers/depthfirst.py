import copy

from env import BaitEnv


class DFSAgent:
    def __init__(self, env :BaitEnv, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed
        self.closed=[] #记录已经走过的状态（state）
        self.action=[] #记录目标action_id

    def dfs(self,env):

        if self.tick > self.tick_max:
            assert 0

        env_copy=copy.deepcopy(env)
        actions=env_copy.action_space
        for action_id in actions:
            next_state, reward, isOver, info = env_copy.step(action_id)
            if next_state in self.closed: #这边走过了或者是撞墙上了或者是推不动，应该是吧。
                continue #直接进入下一个方向
            self.tick += 1 #没有走过，就步数加一
            if isOver and info['message'] != 'Fell into hole. Game over.': #游戏结束并且没掉洞里（成功）
                self.action.append(action_id) #把这步加入目标
                return True
            else:   #掉洞里或者撞墙或者就是普通的没有成功在路上
                if isOver : #掉洞里了
                    return False #返回到上一个函数，然后就是删掉这一步。
                else: #在路上
                    self.closed.append(next_state) #这个新状态加进去
                    self.action.append(action_id)  #这一步加进去
                    if self.dfs(self,next_state):
                        return True
                    self.action.pop() #如果接下来的都返回False，说明这一步走不通，删掉。
        print(f"Used steps: {self.tick} / {self.tick_max}")
        return  False

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        #actions = self.env.action_space
        action_sequence = self.dfs(state)
        return action_sequence