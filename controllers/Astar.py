class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.action=[]
        self.open=[]
        self.close=[]

    def astar(self):

        if self.tick > self.tick_max:
            assert 0

        raise NotImplementedError

    def solve(self):
        self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        actions.pop(0)
        if self.astar():
            action_sequence = self.action
        return action_sequence
    def cost(self):
        pass

    def inspire(self):
        pass
    def act(self, env):
        
        raise NotImplementedError