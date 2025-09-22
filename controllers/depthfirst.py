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

    def dfs(self,env,actions):

        if self.tick > self.tick_max:
            assert 0

        for action_id in actions:
            env_copy=copy.deepcopy(env)
            next_state, reward, isOver, info = env_copy.step(action_id)
            if next_state in self.closed: #这边走过了或者是撞墙上了或者是推不动，应该是吧。
                continue #直接进入下一个方向
            self.tick += 1 #没有走过，就步数加一
            print(f"Used steps: {self.tick} / {self.tick_max}")
            if isOver and info['message'] != 'Fell into hole. Game over.': #游戏结束并且没掉洞里（成功）
                self.action.append(action_id) #把这步加入目标
                return True
            else:   #掉洞里或者撞墙或者就是普通的没有成功在路上//或者是没钥匙单撞门了
                if isOver : #掉洞里了
                    return False #返回到上一个函数，然后就是删掉这一步。
                elif info != {} and info['message'] =="Need key to open goal":#撞墙不会产生新状态，但没钥匙撞门会。
                    self.closed.append(next_state) #加入新状态
                    continue #处理和走入旧状态是一样的
                else: #在路上
                    self.closed.append(next_state) #这个新状态加进去
                    self.action.append(action_id)  #这一步加进去
                    if self.dfs(env_copy,actions):
                        return True
                    self.action.pop()
                    self.tick -= 1  #如果接下来的都返回False，说明这一步走不通，删掉。
        return  False

    def solve(self):
        self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        actions.pop(0)
        if self.dfs(self.env, actions):
            action_sequence = self.action
            return action_sequence
        return None

    def act(self, env):
        raise NotImplementedError