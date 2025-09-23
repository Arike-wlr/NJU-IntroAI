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

    def limiteddfs(self, env):
        opt_grade = 1e8
        sign = False  # 用于标记当前节点是否有可行的子节点,最初设为没有
        for action_id in self.env.action_space:
            env_copy = copy.deepcopy(env)
            state, reward, isOver, info = env_copy.step(action_id)  # 进行一步操作
            self.tick += 1

            if state in self.closed or info == {'message': 'Fell into hole. Game over.'}:
                # 1.重复,退回去，下一步；与2.死掉，退回去，下一步。
                self.tick -= 1
                continue

            sign = True  # 这一步没有重复也没有死掉，没问题。
            self.closed.append(state)
            if isOver or self.tick > self.tick_max:
                # 3.赢了，使用启发式函数评估当前状态，4.超了，使用启发式函数评估当前状态。
                opt_grade = min(opt_grade, self.heuristic(state, reward))
            else:  # 5.在路上，递归调用自身继续搜索。
                opt_grade = min(opt_grade, self.limiteddfs(env_copy))

        self.tick -= 1  # 回退步数计数器（回溯）
        if not sign:
            return 1e8  # 如果没有有效子节点，返回极大值表示无效路径
        else:
            return opt_grade  # 否则返回找到的最优代价

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
            grade = min(self.heuristic(state, reward), self.limiteddfs(env_copy))

            if opt_grade > grade:
                opt_grade = grade
                opt_action = action_id

        return opt_action

