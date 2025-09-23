import copy
class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.closed = []

    @staticmethod
    def find_value(arr, target):
        return [(i, j) for i, row in enumerate(arr)
                for j, cell in enumerate(row)
                if target in cell]

    def heuristic(self, state, reward):
        bias = self.tick  # 用时尽可能少
        bias -= reward  # 尽可能多得奖励

        K_indices = self.find_value(state, 'key')
        G_indices = self.find_value(state, 'goal')
        AN_indices = self.find_value(state, "avatar_nokey")
        AW_indices = self.find_value(state, 'avatar_withkey')
        if AN_indices or AW_indices: # 人活着
            if K_indices: # 有钥匙，还没拿到
                A_index = AN_indices[0]
                K_index = K_indices[0]
                G_index = G_indices[0]
                distance = abs(K_index[0] - A_index[0]) + abs(K_index[1] - A_index[1]) + abs(K_index[0]-G_index[0])+abs(K_index[0]-G_index[1])
            elif G_indices: # 没钥匙，有门
                A_index = AW_indices[0]
                G_index = G_indices[0]
                distance = abs(G_index[0] - A_index[0]) + abs(G_index[1] - A_index[1])
            else: #没钥匙，没门，成功
                distance = 0
        else: # 人没了
            distance = 1e8
        return distance+bias

    def limiteddfs(self, env,actions): # 用法 新 *
            best_grade = 1e8  # 跟踪最大值
            valid_flag = False
            for action_id in actions:
                env_copy = copy.deepcopy(env)
                next_state, reward, isOver, info = env_copy.step(action_id)
                # 检查是否已访问过该状态
                if next_state in self.closed:  # 重复
                    continue
                if 'Cannot push box' in info.get('message', ''):
                    continue
                valid_flag = True
                self.tick += 1
                if isOver:
                    if info.get('message') != 'Fell into hole. Game over.':  # 成功
                        self.closed.append(next_state)
                        grade = self.heuristic(next_state, reward)
                    else:  # 死掉
                        self.tick -= 1
                        valid_flag = False
                        continue
                elif self.tick > self.tick_max:  # 超过最大步数
                    self.closed.append(next_state)
                    grade = self.heuristic(next_state, reward)
                elif info != {} and info.get('message') == "Need key to open goal":  # 没钥匙撞门
                    self.closed.append(next_state)
                    self.tick -= 1
                    continue
                else:  # 在路上
                    self.closed.append(next_state)
                    grade = self.limiteddfs(env_copy, actions)

                best_grade = min(best_grade, grade)
            if not valid_flag:
                return 1e8
            else:
                return best_grade

    def act(self, env):
        """ 进行一次搜索，返回最优的方向 """
        opt_grade = 1e8  # 初始化，最大
        opt_action = 0  # 初始化,不动
        closed = []

        for action_id in [1,2,3,4]:
            env_copy = copy.deepcopy(env)
            state, reward, isOver, info = env_copy.step(action_id)

            if state in closed or info == {'message': 'Fell into hole. Game over.'}:
                continue

            closed.append(state)

            self.closed = []
            self.tick = 0
            grade = min(self.heuristic(state, reward), self.limiteddfs(env_copy,self.env.action_space))

            if opt_grade > grade:
                opt_grade = grade
                opt_action = action_id

        return opt_action

