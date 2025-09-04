import copy
from env import BaitEnv
class LimitedDFSAgent:
    def __init__(self, env:BaitEnv, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.closed=[]
        self.action=[]

    def limiteddfs(self,env,actions):

        if self.tick > self.tick_max:
            assert 0

        action_with_distance={}
        for action_id in actions:
            env_copy = copy.deepcopy(env) #每次copy的都是原来传进来的env对吧
            next_state, reward, isOver, info = env_copy.step(action_id)
            action_with_distance[action_id]=self.calculate_distance(next_state)
        sorted_dict = dict(sorted(action_with_distance.items(), key=lambda item: item[1]))
        sorted_actions=list(sorted_dict.keys())

        for action_id in sorted_actions:
            env_copy = copy.deepcopy(env)
            next_state, reward, isOver, info = env_copy.step(action_id)
            if next_state in self.closed:
                continue
            self.tick += 1
            print(f"Used steps: {self.tick} / {self.tick_max}")
            if isOver and info['message'] != 'Fell into hole. Game over.':
                self.action.append(action_id)
                return True
            else:
                if isOver:
                    return False
                elif info != {} and info['message'] == "Need key to open goal":
                    self.closed.append(next_state)
                    continue
                else:
                    self.closed.append(next_state)
                    self.action.append(action_id)
                    if self.limiteddfs(env_copy, actions):
                        return True
                    self.action.pop()
                    self.tick -= 1
        return False

    def solve(self):
        self.env.reset()
        actions = self.env.action_space
        actions.pop(0)
        if self.limiteddfs(self.env,actions):
            action_sequence = self.action
            return action_sequence
        return None

    def calculate_distance(self,state):
        K_indice=self.find_value(state,'key')
        G_indice=self.find_value(state,'goal')
        if K_indice:
            A_index=self.find_value(state,"avatar_nokey")[0]
            K_index=K_indice[0]
            distance=abs(K_index[0]-A_index[0])+abs(K_index[1]-A_index[1])
        elif G_indice:
            A_index=self.find_value(state,'avatar_withkey')[0]
            G_index=self.find_value(state,'goal')[0]
            distance = abs(G_index[0] - A_index[0]) + abs(G_index[1] - A_index[1])
        else:
            distance=0
        return distance

    @staticmethod
    def find_value(arr, target):
        return [(i, j) for i, row in enumerate(arr)
                for j, cell in enumerate(row)
                if target in cell]

    def act(self, env):
        
        raise NotImplementedError