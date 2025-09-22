import copy
from env import BaitEnv
from math import inf
class LimitedDFSAgent:
    def __init__(self, env:BaitEnv, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.closed=[]

    def limiteddfs(self,env,actions):
        best_grade=1e8 #跟踪最大值
        valid_flag=False
        for action_id in actions:
            env_copy = copy.deepcopy(env)
            next_state, reward, isOver, info = env_copy.step(action_id)
            # 检查是否已访问过该状态
            if next_state in self.closed: #重复
                continue
            if 'Cannot push box' in info.get('message', ''):
                continue
            valid_flag=True
            self.tick += 1
            if isOver:
                if info.get('message') != 'Fell into hole. Game over.': # 成功
                    self.closed.append(next_state)
                    grade = self.heuristic(next_state,reward)
                else: # 死掉
                    self.tick -= 1
                    valid_flag=False
                    continue
            elif self.tick >self.tick_max: # 超过最大步数
                self.closed.append(next_state)
                grade = min(best_grade,self.heuristic(next_state, reward))

            elif info != {} and info.get('message') == "Need key to open goal": # 没钥匙撞门
                valid_flag=False
                self.closed.append(next_state)
                self.tick -= 1
                continue

            else:# 在路上
                self.closed.append(next_state)
                grade = min(best_grade,self.limiteddfs(env_copy, actions))

            best_grade = min(best_grade,grade)
        if not valid_flag:
            return 1e8
        else:
            return best_grade

    def heuristic(self,state,reward):
        K_indices = self.find_value(state, 'key')
        G_indices = self.find_value(state, 'goal')
        AN_indices = self.find_value(state, "avatar_nokey")
        AW_indices = self.find_value(state, 'avatar_withkey')
        bias = self.tick  # 用时尽可能少
        bias -= reward  # 尽可能多得奖励

        if not (AN_indices or AW_indices):
            return inf

        if AW_indices:  # 已经有钥匙
            if G_indices:  # 有目标位置
                A_pos = AW_indices[0]
                G_pos = G_indices[0]
                return abs(G_pos[0] - A_pos[0]) + abs(G_pos[1] - A_pos[1])+bias
            else:
                return inf  # 有钥匙但没目标，不是有效状态
        else:  # 没有钥匙
            if K_indices:  # 有钥匙可拿
                A_pos = AN_indices[0]
                K_pos = K_indices[0]
                key_dist = abs(K_pos[0] - A_pos[0]) + abs(K_pos[1] - A_pos[1])
                # 估计拿到钥匙后到最近目标的距离
                if G_indices:
                    G_pos = G_indices[0]
                    goal_dist = abs(G_pos[0] - K_pos[0]) + abs(G_pos[1] - K_pos[1])
                    return key_dist + goal_dist+bias
                return key_dist+bias
            else:
                return inf  # 没钥匙也没目标

    @staticmethod
    def find_value(arr, target):
        return [(i, j) for i, row in enumerate(arr)
                for j, cell in enumerate(row)
                if target in cell]

    def act(self, env):
        """走一步，搜索一次。"""
        opt_grade = 1e8  # 初始化，最大
        opt_action = 0 # 初始化，不动
        closed=[]

        for action_id in [1,2,3,4]:
            env_copy = copy.deepcopy(env)
            state, reward, isOver, info = env_copy.step(action_id)

            if state in closed or info == {'message': 'Fell into hole. Game over.'}: # 合并重复和掉洞。
                continue
            if 'Cannot push box' in info.get('message', ''):
                continue

            closed.append(state)
            self.closed=[]
            self.tick = 0
            grade = min(self.heuristic(state, reward), self.limiteddfs(env_copy,[1,2,3,4]))
            if opt_grade > grade:
                opt_grade = grade
                opt_action = action_id

        return opt_action