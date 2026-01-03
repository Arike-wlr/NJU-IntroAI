import os

os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow as tf
from environment.go import Position
from environment import coords
from opponent_pool import OpponentPool
from agent.agent import GoPolicyAgent
import json


def play_game_between_agents(agent1, agent2=None, agent1_is_black=True):
    """两个agent对弈"""
    game = Position(komi=0.5)

    while not game.is_game_over():
        current_player = 0 if game.to_play == 1 else 1

        if (current_player == 0 and agent1_is_black) or \
                (current_player == 1 and not agent1_is_black):
            # agent1走棋
            action, _ = agent1.select_action(game, is_evaluation=True)
        else:
            # agent2走棋或随机
            if agent2:
                action, _ = agent2.select_action(game, is_evaluation=True)
            else:
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) == 0:
                    action = 25
                else:
                    action = np.random.choice(legal_actions)

        # 执行动作
        try:
            if action == 25:
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                game = game.play_move(point, mutate=False)
        except Exception as e:
            print(f"执行动作 {action} 时出错: {e}")
            # 选择合法动作
            legal_moves = game.all_legal_moves()
            legal_actions = np.where(legal_moves == 1)[0]
            if len(legal_actions) > 0:
                action = legal_actions[0]
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)

    result = game.result()
    # agent1视角的结果
    if agent1_is_black:
        return result  # 黑棋胜1，白棋胜-1
    else:
        return -result


def create_optimized_agents():
    """创建使用优化参数的agent"""
    # 加载优化参数
    try:
        with open("./bayesian_optimization/best_params.json", 'r') as f:
            best_params = json.load(f)
        params = best_params['params']

        print("使用贝叶斯优化参数:")
        print(f"  Critic学习率: {params.get('critic_lr', 0.01):.6f}")
        print(f"  Policy学习率: {params.get('pi_lr', 0.001):.6f}")
        print(f"  熵正则化: {params.get('entropy_cost', 0.01):.4f}")
        print(f"  批次大小: {params.get('batch_size', 32)}")
        print(f"  Critic更新次数: {params.get('num_critic_before_pi', 8)}")
    except FileNotFoundError:
        print("警告: 未找到优化参数文件，使用默认参数")
        params = {
            'critic_lr': 0.01,
            'pi_lr': 0.001,
            'entropy_cost': 0.01,
            'batch_size': 32,
            'num_critic_before_pi': 8
        }

    # 创建深度网络
    tf.reset_default_graph()
    deep_sess = tf.Session()
    deep_agent = GoPolicyAgent(
        session=deep_sess,
        hidden_layers=[256, 256],
        loss_str="a2c"
    )

    # 应用优化参数
    deep_agent.agent._critic_learning_rate = params.get('critic_lr', 0.01)
    deep_agent.agent._pi_learning_rate = params.get('pi_lr', 0.001)
    deep_agent.agent._entropy_cost = params.get('entropy_cost', 0.01)
    deep_agent.agent._batch_size = params.get('batch_size', 32)
    deep_agent.agent._num_critic_before_pi = params.get('num_critic_before_pi', 8)

    deep_sess.run(tf.global_variables_initializer())

    # 创建浅层网络
    tf.reset_default_graph()
    rollout_sess = tf.Session()
    rollout_agent = GoPolicyAgent(
        session=rollout_sess,
        hidden_layers=[64],
        loss_str="a2c"
    )
    rollout_sess.run(tf.global_variables_initializer())

    return deep_agent, rollout_agent, deep_sess, rollout_sess


def train_with_opponent_pool_optimized():
    """使用优化参数的对手池训练"""
    print("=" * 70)
    print("优化参数对手池训练")
    print("=" * 70)

    # 初始化对手池
    opponent_pool = OpponentPool(pool_dir="./opponent_pool_optimized", max_size=8)

    # 训练参数
    total_iterations = 10
    games_per_iteration = 20
    eval_games = 10

    # 使用优化参数创建agent
    print("创建优化参数网络...")
    deep_agent, rollout_agent, deep_sess, rollout_sess = create_optimized_agents()

    # 如果对手池为空，添加初始模型
    if len(opponent_pool.opponents) == 0:
        print("对手池为空，创建初始模型...")

        # 评估初始模型
        print("评估初始模型...")
        eval_wins = 0
        for i in range(5):
            result = play_game_between_agents(deep_agent, None)
            if result > 0:
                eval_wins += 1

        initial_win_rate = eval_wins / 5

        # 添加到对手池
        opponent_pool.add_double_agent(
            deep_agent, rollout_agent,
            name=f"optimized_initial_wr{initial_win_rate:.2f}",
            win_rate=initial_win_rate
        )

        print(f"创建初始模型，胜率: {initial_win_rate:.2%}")

    iteration_stats = []

    for iteration in range(total_iterations):
        print(f"\n{'=' * 60}")
        print(f"第 {iteration + 1}/{total_iterations} 轮迭代")
        print('=' * 60)

        # 打印对手池状态
        opponent_pool.print_pool_status()

        # 1. 从对手池选择对手
        if iteration < 3:
            opponent_name = opponent_pool.get_opponent(strategy="weakest")
        else:
            opponent_name = opponent_pool.get_opponent(strategy="balanced")

        # 加载对手
        opp_deep_agent = None
        opp_deep_sess = None

        if opponent_name:
            opp_deep_agent, opp_deep_sess = opponent_pool.load_agent(opponent_name, model_type="deep")

            if opp_deep_agent:
                elo = opponent_pool.elo_ratings.get(opponent_name, 1500)
                print(f"对战对手: {opponent_name} (Elo: {elo:.0f})")
            else:
                opponent_name = None
        else:
            opponent_name = None

        # 2. 进行训练对局
        print(f"开始训练对局...")
        wins = 0
        results = []

        for game_idx in range(games_per_iteration):
            # 交替黑白棋
            agent_is_black = (game_idx % 2 == 0)

            if opponent_name and opp_deep_agent:
                # 与对手池中的agent对弈
                result = play_game_between_agents(
                    deep_agent, opp_deep_agent,
                    agent1_is_black=agent_is_black
                )
            else:
                # 与随机对手对弈
                result = play_game_between_agents(deep_agent, None, agent1_is_black=agent_is_black)

            results.append(result)
            if (agent_is_black and result > 0) or (not agent_is_black and result < 0):
                wins += 1

            # 显示进度
            if (game_idx + 1) % 5 == 0:
                print(f"  完成 {game_idx + 1}/{games_per_iteration} 局")

        # 3. 计算胜率
        win_rate = wins / games_per_iteration
        avg_result = np.mean(results)

        print(f"训练结果:")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  平均结果: {avg_result:.3f}")

        # 4. 更新Elo（如果是对手池中的对手）
        if opponent_name and opp_deep_agent:
            current_agent_name = f"optimized_iter{iteration:03d}"

            # 计算Elo结果
            if win_rate > 0.6:
                result_score = 1.0
            elif win_rate < 0.4:
                result_score = 0.0
            else:
                result_score = 0.5

            opponent_pool.update_elo(
                current_agent_name,
                opponent_name,
                result=result_score,
                k=32
            )

            # 关闭对手会话
            if opp_deep_sess:
                opp_deep_sess.close()

        # 5. 将当前agent添加到对手池
        agent_name = f"optimized_iter{iteration:03d}_wr{win_rate:.2f}"
        print(f"保存当前模型到对手池: {agent_name}")

        opponent_pool.add_double_agent(
            deep_agent, rollout_agent,
            name=agent_name,
            win_rate=win_rate
        )

        # 6. 评估当前agent性能
        print(f"评估当前agent性能...")
        eval_wins = 0
        eval_results = []

        for i in range(eval_games):
            result = play_game_between_agents(deep_agent, None)
            eval_results.append(result)
            if result > 0:
                eval_wins += 1

        eval_win_rate = eval_wins / eval_games
        eval_avg_result = np.mean(eval_results)

        print(f"评估结果:")
        print(f"  胜率: {eval_win_rate:.2%}")
        print(f"  平均结果: {eval_avg_result:.3f}")

        # 7. 保存检查点
        print(f"保存检查点...")
        os.makedirs("./saved_models_optimized/opponent_pool", exist_ok=True)
        deep_agent.save(f"./saved_models_optimized/opponent_pool/deep_iteration_{iteration}")
        rollout_agent.save(f"./saved_models_optimized/opponent_pool/rollout_iteration_{iteration}")

        iteration_stats.append({
            'iteration': iteration,
            'train_win_rate': win_rate,
            'eval_win_rate': eval_win_rate,
            'eval_avg_result': eval_avg_result,
            'opponent': opponent_name if opponent_name else "random"
        })

    # 保存最终模型
    print(f"保存最终模型...")
    deep_agent.save("./saved_models_optimized/opponent_pool/deep_policy_final")
    rollout_agent.save("./saved_models_optimized/opponent_pool/rollout_policy_final")

    for stat in iteration_stats:
        print(f"Iteration {stat['iteration']:2d}: "
              f"训练胜率={stat['train_win_rate']:5.1%} | "
              f"评估胜率={stat['eval_win_rate']:5.1%} | "
              f"平均结果={stat['eval_avg_result']:6.3f} | "
              f"对手={stat['opponent']}")

    # 打印对手池最终状态
    opponent_pool.print_pool_status()

    # 关闭会话
    deep_sess.close()
    rollout_sess.close()

    return deep_agent, rollout_agent, iteration_stats


if __name__ == "__main__":
    import shutil
    import os
    if os.path.exists("./opponent_pool_optimized"):
        shutil.rmtree("./opponent_pool_optimized")
        print("删除旧对手池")

    if os.path.exists("./saved_models_optimized/opponent_pool"):
        shutil.rmtree("./saved_models_optimized/opponent_pool")
        print("删除旧模型")

    os.makedirs("./saved_models_optimized", exist_ok=True)

    # 运行优化参数训练
    deep_agent, rollout_agent, stats = train_with_opponent_pool_optimized()

    print("优化参数训练完成！")