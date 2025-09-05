import copy
from math import inf

class Node:
    def __init__(self,state,g,father=None,action_id=None):
        self.state=state
        self.g=g
        self.h=self.Get_h(self.state)
        self.f=self.h+self.g
        self.father=father # father直接存储另一个 Node对象时，自然就是“指针”行为
        self.action_id=action_id

    def Get_h(self,state):
        K_indice = self.find_value(state, 'key')
        G_indice = self.find_value(state, 'goal')
        AN_indice = self.find_value(state, "avatar_nokey")
        AW_indice = self.find_value(state, 'avatar_withkey')
        if AN_indice or AW_indice:
            if K_indice:
                A_index = AN_indice[0]
                K_index = K_indice[0]
                distance = abs(K_index[0] - A_index[0]) + abs(K_index[1] - A_index[1])
            elif G_indice:
                A_index = AW_indice[0]
                G_index = G_indice[0]
                distance = abs(G_index[0] - A_index[0]) + abs(G_index[1] - A_index[1])
            else:
                distance = 0
        else:
            distance = inf
        return distance

    @staticmethod
    def find_value(arr, target):
        return [(i, j) for i, row in enumerate(arr)
                for j, cell in enumerate(row)
                if target in cell]

class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.action=[]
        initial,reward,isover,info=env.step(0)
        self.open=[Node(initial,0)]
        self.close=[]

    def astar(self,env,actions):

        if self.tick > self.tick_max:
            assert 0

        while self.open:
            curr=self.find_least_cost(self.open)
            print(curr.h)
            self.close.append(curr)
            self.open.remove(curr)

            if curr.h==0:
                action=[]
                while curr.father != None:
                    action.append(curr.action_id)
                    curr=curr.father
                self.action=action[::-1]
                return True

            for action_id in actions:
                env_copy=copy.deepcopy(env)
                next_state, reward, isOver, info = env_copy.step(action_id) #遍历当前节点的邻居节点
                currnode=Node(next_state,curr.g+1,curr,action_id) #将邻居节点存成Node类型。
                exist = False
                for node in self.close:
                    if next_state == node.state:
                        exist=True#如果该节点在close列表中
                        break #不用遍历close列表了
                if exist:
                    continue #然后忽略它

                self.tick += 1
                print(f"Used steps: {self.tick} / {self.tick_max}")

                exist = False
                for i in range(len(self.open)):
                    if next_state == self.open[i].state: #如果该节点在open列表中
                        exist = True
                        if currnode.g < self.open[i].g: #如果产生更小的g值
                            self.open[i]=currnode
                        break
                if not exist:
                    self.open.append(currnode)

        return False

    def solve(self):
        self.env.reset()
        actions = self.env.action_space
        actions.pop(0)
        if self.astar(self.env,actions):
            action_sequence = self.action
            return action_sequence
        return None

    @staticmethod
    def find_least_cost(nodes):
        f=[node.f for node in nodes]
        return nodes[f.index(min(f))]

    def inspire(self):
        pass
    def act(self, env):
        
        raise NotImplementedError