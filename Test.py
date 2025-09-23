import copy
class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.init_state()
        self.tick_max = tick_max
        self.tick = 0
        self.closed = list()

    def init_state(self):
        """ 得到钥匙、终点的位置坐标 """
        state = self.env.reset()
        self.row_len = len(state)
        self.col_len = len(state[0])
        for i in range(self.row_len):
            for j in range(self.col_len):
                if 'key' in state[i][j]:
                    self.key_pos = (i, j)
                if 'goal' in state[i][j]:
                    self.goal_pos = (i, j)

    def get_pos_and_state(self, state):
        """ 得到精灵位置以及精灵状态 """
        for i in range(self.row_len):
            for j in range(self.col_len):
                # 有钥匙返回True，没钥匙返回False
                if 'avatar_nokey' in state[i][j]:
                    return i, j, False
                if 'avatar_withkey' in state[i][j]:
                    return i, j, True

    def heuristic(self, state, reward) -> int:  # 计算结果越小，说明情况越好
        """ 计算启发式函数得分 """
        pos_x, pos_y, withkey = self.get_pos_and_state(state)

        bias = self.tick  # 倾向于用时尽可能少
        bias -= reward  # 倾向于尽可能多得奖励

        if withkey:  # 有钥匙，则计算精灵与门口的距离
            aim_x, aim_y = self.goal_pos
        else:  # 无钥匙，则计算精灵与钥匙的距离加钥匙与门口的距离
            aim_x, aim_y = self.key_pos
            bias += abs(self.key_pos[0] - self.goal_pos[0]) + \
                    abs(self.key_pos[1] - self.goal_pos[1])

        return abs(pos_x - aim_x) + abs(pos_y - aim_y) + bias

    # 修改depthfirst的dfs代码，改为计算并返回opt_grade
    def limiteddfs(self, env, opt_grade=1e8):  # （默认值为很大的数1e8）
        """ 有限制的深度优先，搜索到终点的最短路径
        	返回值：找到的最优路径的代价!!!!"""
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
                print("win",opt_grade)
            else:  # 5.在路上，递归调用自身继续搜索。
                opt_grade = min(opt_grade, self.limiteddfs(env_copy))
                print("on",opt_grade)

        self.tick -= 1  # 回退步数计数器（回溯）
        if not sign:
            return 1e8  # 如果没有有效子节点，返回极大值表示无效路径
        else:
            return opt_grade  # 否则返回找到的最优代价

    def act(self, env):
        """ 进行一次搜索，返回最优的方向 """
        opt_action = 0  # 一个动作
        opt_grade = 1e8  # 初始化，最大
        closed = list()  # /[]

        # 探索四个方向哪个方向“最优”：
        for action_id in self.env.action_space:
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

