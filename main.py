import gym
from collections import deque
import random
import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from agent import DQNAgent, DDQNAgent

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="dqn")
    parser.add_argument("--num_episodes", type=int, default=600)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_frequency", type=int, default=10)
    parser.add_argument("--save_plots",action="store_true", default=True, help="是否保存图像")

    args = parser.parse_args()
    return args


def eval_policy(agent):
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = agent.act(state, eps=0.)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    # print(f"Return {return_}") 
    return return_


def train(args, agent, buffer):
    episodes = []
    epi_loss = []
    eval_returns = []  # 记录评估时的return
    # Training loop
    for episode in range(args.num_episodes):
        # Reset the environment
        state = env.reset()
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay_rate ** episode))

        # Run one episode
        losses = []
        return_ = 0
        for step in range(args.max_steps_per_episode):
            # Choose and perform an action
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            buffer.append((state, action, reward, next_state, done))
            
            if len(buffer) >= args.batch_size:
                batch = random.sample(buffer, args.batch_size)
                # Update the agent's knowledge
                loss = agent.learn(batch, args.gamma)
                losses.append(loss)
            return_ += reward
            
            state = next_state
            
            # Check if the episode has ended
            if done:
                break
        episodes.append(episode+1)
        loss = torch.mean(torch.tensor(losses))
        epi_loss.append(loss)
        eval_return = eval_policy(agent)
        eval_returns.append(eval_return)

        print(f"Episode {episode + 1} Step {step + 1}: Training Loss {loss}, Return {eval_return}")
    plot_training_progress(episodes, losses, eval_returns, args)

def plot_training_progress(episodes, losses, eval_returns, args):
    """绘制训练进度图像"""
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制回报曲线
    ax1.plot(episodes, eval_returns, 'b-', alpha=0.6, label='Evaluation Return')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title(f'{args.agent_name.upper()} - Evaluation Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制损失曲线
    if losses and len(losses) > 0:
        # 确保losses是纯Python数值列表
        losses_clean = [float(loss) for loss in losses]
        ax2.plot(episodes[:len(losses_clean)], losses_clean, 'g-', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{args.agent_name.upper()} - Training Loss')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    if args.save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{args.agent_name}_training_progress_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"训练图像已保存为: {plot_filename}")

    plt.show()

def load_optimized_params(agent_name='dqn'):
    """加载优化后的参数"""
    param_file = f"best_params_{agent_name}.json"
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            return json.load(f)
    else:
        print(f"警告: 未找到优化参数文件 {param_file}，使用默认参数")
        return None

if __name__ == "__main__":
    best_params = load_optimized_params()

    if best_params is None:
        # 如果没有找到最佳参数文件，使用默认参数
        args = parser()
    else:
        # 使用最佳参数
        args = argparse.Namespace()
        args.agent_name = "dqn"
        args.num_episodes = 600  # 最终训练可以用更多episode
        args.max_steps_per_episode = 500
        args.epsilon_start = best_params["epsilon_start"]
        args.epsilon_end = best_params["epsilon_end"]
        args.epsilon_decay_rate = best_params["epsilon_decay_rate"]
        args.gamma = best_params["gamma"]
        args.lr = best_params["lr"]
        args.buffer_size = best_params["buffer_size"]
        args.batch_size = best_params["batch_size"]
        args.update_frequency = best_params["update_frequency"]
        args.save_plots=True

    print("Using parameters:", vars(args))

    # Set up the environment
    env = gym.make("CartPole-v1")

    buffer = deque(maxlen=args.buffer_size)

    # Initialize the DQNAgent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    if args.agent_name == "dqn":
        agent = DQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)
    elif args.agent_name == "ddqn":
        agent = DDQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)
    else:
        assert False, "Not Implement agent!"

    train(args, agent, buffer)
    env.close()