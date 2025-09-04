import copy
import  argparse
from env import BaitEnv
import logging
import sys
from controllers.random import RandomAgent
from controllers.depthfirst import DFSAgent
from controllers.limitdepthfirst import LimitedDFSAgent
from controllers.Astar import AstarAgent
from controllers.MCTS import MCTSAgent

if __name__ == "__main__":
    
    print("Game start!")
    level = 0 #第一关
    env = BaitEnv(level=level, render=False)
    
    # actions: 0 noop, 1 left, 2 right, 3 down, 4 up
    parser = argparse.ArgumentParser(description="Bait 游戏，请选择执行模式")
    parser.add_argument(
        "--mode",
        choices=["random" , "play", "depthfirst", "limitdepthfirst", "Astar", "MCTS"],
        required=True,
        help="运行模式:random--随机运行；depthfirst--深度优先搜索；limitdepthfirst--"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    action_lst = None
    if args.mode == "play":
        # input your own actions here
        tick_max = 30
        action_lst = [3, 2, 3, 1, 3, 4, 4, 4, 1, 0]
    elif args.mode == "random":
        tick_max = 30
        agent = RandomAgent(env, tick_max)
    elif args.mode == "depthfirst":
        tick_max = 10000
        agent = DFSAgent(env, tick_max)
        action_lst = agent.solve()
    elif args.mode == "limitdepthfirst":
        tick_max = 100
        agent = LimitedDFSAgent(env, tick_max)
        action_lst = agent.solve()
    elif args.mode == "Astar":
        tick_max = 100
        agent = AstarAgent(env, tick_max)
    elif args.mode == "MCTS":
        tick_max = 1000
        agent = MCTSAgent(env, tick_max)
    else:
        raise ValueError(f"未知模式: {args.mode}")

    print("Action list:", action_lst)
    action_lst_len = len(action_lst) if action_lst else 1e8

    env = BaitEnv(level=level, render=True)
    env.reset()
    for step in range(min(30, action_lst_len)):
        if action_lst:
            action_id = action_lst[step]
        else:
            env_copy = copy.deepcopy(env)
            env_copy.render = False
            action_id = agent.act(env_copy)
        state, reward, isOver, info = env.step(action_id)
        print(f"Step: {step}, Action taken: {action_id}, Reward: {reward}, Done: {isOver}, Info: {info}")
        if isOver:
            break

    env.make_gif()