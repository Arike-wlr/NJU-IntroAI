import random
import math
import copy

ACTIONS = [1, 2, 3, 4]

class Node:
    def __init__(self, move='', env='', parent=None):
        self.move = move
        self.parent = parent # 父节点
        self.children = [] # 子节点
        self.visits = 0 # 访问次数
        self.score = 0
        self.env = env
        if parent is None:
            self.state = []
        else:
            self.state = self.parent.state + [move]

    def is_fully_expanded(self):

        """判断是否四个方向都尝试过一遍了。"""

        return len(self.children) == len(ACTIONS)

    def expand(self):

        """随机选一个没有尝试过的方向尝试，并把结果加入该节点的children参数，返回该子节点。"""

        moves_tried = [child.move for child in self.children]
        untried_moves = [move for move in ACTIONS if move not in moves_tried]
        move = random.choice(untried_moves)
        env_child = copy.deepcopy(self.env)
        env_child.step(move)
        child_node = Node(move=move, env=env_child, parent=self)
        self.children.append(child_node)
        return child_node

    def simulate(self):

        """随机一步步向前走"""

        if self.env.done:#如果此时游戏结束，goal存在就是死了，返回（0分，用1步），goal不存在就是赢了，返回（5分，用1步）。
            return (0, 1) if self.env.goal_exists else (5, 1)
        #如果还没结束
        env = copy.deepcopy(self.env)
        score = 0
        done = False
        used_ticks = 0
        while not done:#循环
            action = random.choice(ACTIONS)
            _, reward, done, _ = env.step(action)
            used_ticks += 1
            score += reward
            if done or len(self.state) + used_ticks > 20:
                break #游戏结束或者走过的路超过了20步，就停止。
        return score, used_ticks # 返回得分和使用的步数

    def backpropagate(self, result):
        self.visits += 1 # 访问次数加1
        self.score += result # 累计得分增加，从这个节点出发获得了多少总分
        if self.parent:
            self.parent.backpropagate(result) # 递归向上传播

    def print_tree(self, indent=0):
        """如名字所说，打印树"""
        prefix = '    ' * indent
        print(f"{prefix}Move: {self.move}, Visits: {self.visits}, Score: {self.score}")
        for child in self.children:
            child.print_tree(indent + 1)

class MCTS:
    def __init__(self, tick_max, env):
        self.root = Node(move=None, env=env)
        self.tick_max = tick_max

    def select(self):
        """选择下一个访问（用于扩展）的节点，（若当前节点未完全扩展，就返回当前节点）"""
        node = self.root # 从根节点开始，根据ucb1返回的值，
        while node.is_fully_expanded(): # 在当前节点充分扩展的情况下
            node = max(node.children, key=self.ucb1) # 使用UCB1选择最优子节点
        return node # 否则返回未完全扩展的节点

    def ucb1(self, node):
        """评估方法"""
        Q = node.score / (node.visits + 1e-5) # 平均奖励
        N = node.parent.visits # 父节点访问次数
        n = node.visits # 当前节点访问次数
        c = math.sqrt(2) # 探索系数
        value = Q + c * math.sqrt(math.log(N + 1) / (n + 1e-5)) # 根据ucb公式
        return value + random.random() # 添加随机扰动

    def run(self):
        used_ticks_total = 0 # 记录一共走的步数
        while used_ticks_total < self.tick_max: # 步数没超
            leaf = self.select() # 用于扩展的节点
            if not leaf.is_fully_expanded(): #这个节点没有完全扩展
                leaf = leaf.expand() #扩展它
            score, used_ticks = leaf.simulate() #模拟，每20步回来一次
            leaf.backpropagate(score) #回溯传播
            used_ticks_total += used_ticks #更新总步数，用于判断进不进下一轮循环。

class MCTSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max

    def solve(self):
        self.env.reset()
        self.mcts = MCTS(self.tick_max, self.env) # 构建一个MCTS实例
        self.mcts.run() #模拟跑一次

        # Print the tree after the search （调试用）
        print("MCTS Tree Structure:")
        self.mcts.root.print_tree()

        node = self.mcts.root # 从根往下把得分最大的子节点找出来，形成动作链
        action_sequence = []

        while node.children:
            node = max(node.children, key=lambda node: node.score)
            action_sequence.append(node.move)

        return action_sequence
    
    def act(self, env):
        """实时决策"""
        self.mcts = MCTS(self.tick_max, copy.deepcopy(env)) # 构建一个MCTS实例
        self.mcts.run()

        node = self.mcts.root
        # 选择访问次数最多的子节点
        node = max(node.children, key=lambda node: node.visits)

        return node.move

