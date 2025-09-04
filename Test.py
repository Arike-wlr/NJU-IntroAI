
if __name__=="__main__":
    env = BaitEnv(level=0, render=True)
    env.reset()
    agent = DFSAgent(env, 30)

    initial, reward, isOver, info =env.step(0)
    print(initial)
#    l=agent.calculate_distance(initial)
    #env.reset()
    #initial, reward, isOver, info = env.step(1)
    #print(initial)