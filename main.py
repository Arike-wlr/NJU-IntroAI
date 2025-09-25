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

"""
！！注意！！

我对main.py进行了一点小小的修改，运行时需要输入相应的参数来选择使用的方法和关卡。
运行需要输入：
```cmd
python main.py --mode <运行方法> --level <关卡名>
```
运行方法包括：["random", "play", "depthfirst", "limitdepthfirst", "Astar", "MCTS"]
关卡包括：['0','1','2','3','4']
也可输入 --help 查询：
```cmd
python main.py --help
```
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bait 游戏，请选择执行模式")
    parser.add_argument(
        "--mode",
        choices=["random", "play", "depthfirst", "limitdepthfirst", "Astar", "MCTS"],
        required=True,
        help="运行模式:random--随机运行；depthfirst--深度优先搜索；limitdepthfirst--深度受限的深度优先搜索；Aster--A*算法；MCTS--蒙特卡洛树算法。"
    )
    parser.add_argument(
        "--level",
        choices=['0','1','2','3','4'],
        required=True,
        help="游戏关卡：0~4，共5关。"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    print("Game start!")
    level = int(args.level)
    env = BaitEnv(level=level, render=False)
    
    # actions: 0 noop, 1 left, 2 right, 3 down, 4 up

    action_lst = None
    if args.mode == "play":
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
        tick_max = 30
        agent = LimitedDFSAgent(env, tick_max)
    elif args.mode == "Astar":
        tick_max = 100
        agent = AstarAgent(env, tick_max)
        action_lst = agent.solve()
    elif args.mode == "MCTS":
        tick_max = 1000
        agent = MCTSAgent(env, tick_max)
        action_lst = agent.solve()
    else:
        raise ValueError(f"未知模式: {args.mode}")

    print("Action list:", action_lst)
    action_lst_len = len(action_lst) if action_lst else 1e8

    env = BaitEnv(level=level, render=True)
    env.reset()
    for step in range(min(tick_max, action_lst_len)):
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