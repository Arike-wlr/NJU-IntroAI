import os
os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow as tf
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent

def play_one_game(agent, opponent="random", is_training=True):
    """玩一局游戏，返回经验数据"""
    game = Position(komi=0.5)
    episode_data = []

    while not game.is_game_over():
        current_player = 0 if game.to_play == 1 else 1

        # 获取当前状态
        state = agent.encode_state(game)
        legal_moves = game.all_legal_moves()
        legal_actions = np.where(legal_moves == 1)[0]

        if current_player == 0:  # 我们的AI（黑棋）
            # 选择动作
            if is_training:
                # 训练模式：让agent自己记录transition
                action, _ = agent.select_action(game, is_evaluation=False)
            else:
                # 评估模式
                action, _ = agent.select_action(game, is_evaluation=True)

            # 记录状态（用于手动训练）
            episode_data.append({
                'state': state,
                'action': action,
                'legal_actions': legal_actions
            })

            # 执行动作
            if action == 25:  # PASS
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                game = game.play_move(point, mutate=False)

        else:  # 对手（白棋）
            if opponent == "random":
                # 随机对手
                if len(legal_actions) == 0:
                    action = 25  # PASS
                else:
                    action = np.random.choice(legal_actions)

            elif opponent == "policy":
                # 另一个策略网络对手（用于self-play）
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 25

            # 执行动作
            if action == 25:
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                game = game.play_move(point, mutate=False)

    # 游戏结束，获取结果
    result = game.result()  # 黑胜:1, 白胜:-1, 平:0

    return episode_data, result


def evaluate(agent, num_games=20, opponent="random"):
    """评估agent性能"""
    wins = 0

    for i in range(num_games):
        _, result = play_one_game(agent, opponent=opponent, is_training=False)
        if result > 0:  # 黑棋胜
            wins += 1

    return wins / num_games


def main():
    print("=== 训练深度策略网络 ===")

    # 创建TensorFlow会话
    tf.reset_default_graph()
    sess = tf.Session()

    # 训练深度网络
    print("创建深度网络...")
    deep_agent = GoPolicyAgent(
        session=sess,
        hidden_layers=[256, 256],  # 深层网络
        loss_str="a2c"
    )

    print("训练深度网络...")
    for episode in range(1000):
        # 玩一局游戏（agent会自动记录和学习）
        _, result = play_one_game(deep_agent, opponent="random", is_training=True)

        # 定期评估
        if (episode + 1) % 50 == 0:
            win_rate = evaluate(deep_agent, num_games=10)
            print(f"Episode {episode + 1}/1000, Win Rate: {win_rate:.2f}")

    print("\n=== 训练浅层走子网络 ===")
    # 训练浅层网络
    rollout_agent = GoPolicyAgent(
        session=sess,
        hidden_layers=[64],  # 浅层网络
        loss_str="a2c"
    )

    print("训练浅层网络...")
    for episode in range(500):  # 更少训练
        _, result = play_one_game(rollout_agent, opponent="random", is_training=True)

        if (episode + 1) % 50 == 0:
            win_rate = evaluate(rollout_agent, num_games=10)
            print(f"Episode {episode + 1}/500, Win Rate: {win_rate:.2f}")

    print("\n=== 保存模型 ===")

    print("训练完成！")
    sess.close()


if __name__ == "__main__":
    main()