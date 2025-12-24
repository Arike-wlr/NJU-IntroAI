import os
os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow as tf
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent
from model_saver import ModelSaver


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

    result = game.result()

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
    deep_sess = tf.Session()

    # 创建模型保存器
    model_saver = ModelSaver(save_dir="./saved_models")

    print("=== 训练深层网络 ===")
    deep_agent = GoPolicyAgent(
        session=deep_sess,
        hidden_layers=[256, 256],  # 深层网络
        loss_str="a2c"
    )
    deep_sess.run(tf.global_variables_initializer())
    for episode in range(1000):
        _, result = play_one_game(deep_agent, opponent="random", is_training=True)
        if (episode + 1) % 50 == 0:
            win_rate = evaluate(deep_agent, num_games=10)
            print(f"Episode {episode + 1}/1000, Win Rate: {win_rate:.2f}")

            if (episode + 1) % 100 == 0:
                print(f"  保存检查点...")
                model_saver.save_policy_network(
                    deep_agent,
                    "deep_policy",
                    iteration=episode + 1
                )

    print("=== 训练浅层网络 ===")
    tf.reset_default_graph()
    rollout_sess = tf.Session()
    rollout_agent = GoPolicyAgent(
        session=rollout_sess,
        hidden_layers=[64],
        loss_str="a2c"
    )
    rollout_sess.run(tf.global_variables_initializer())
    for episode in range(500):
        _, result = play_one_game(rollout_agent, opponent="random", is_training=True)
        if (episode + 1) % 50 == 0:
            win_rate = evaluate(rollout_agent, num_games=10)
            print(f"Episode {episode + 1}/500, Win Rate: {win_rate:.2f}")

            if (episode + 1) % 100 == 0:
                print(f"  保存检查点...")
                model_saver.save_policy_network(
                    rollout_agent,
                    "rollout_policy",
                    iteration=episode + 1
                )

    print("=== 保存最终模型 ===")
    model_saver.save_policy_network(deep_agent, "deep_policy_final")
    model_saver.save_policy_network(rollout_agent, "rollout_policy_final")
    print("保存的策略网络:")
    saved_models = model_saver.list_saved_models("policy")
    for model in saved_models:
        print(f"  - {model}")

    deep_sess.close()
    rollout_sess.close()
    print("训练完成！")

if __name__ == "__main__":
    main()