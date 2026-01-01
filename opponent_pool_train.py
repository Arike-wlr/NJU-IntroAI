import os

os.environ['BOARD_SIZE'] = '5'
import numpy as np
from environment.go import Position
from environment import coords
from opponent_pool import OpponentPool
import time


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


def train_with_opponent_pool_simple():
    """简化的对手池训练"""
    print("=== 简化版对手池训练 ===")

    # 初始化对手池
    opponent_pool = OpponentPool(pool_dir="./opponent_pool", max_size=8)

    # 训练参数
    total_iterations = 10
    games_per_iteration = 20
    eval_games = 10

    # 创建初始agent
    from agent.agent import create_policy_agent

    print("\n🔄 创建初始网络...")
    deep_agent, deep_sess = create_policy_agent(
        hidden_layers=[256, 256],
        loss_str="a2c",
        name="current_deep"
    )

    rollout_agent, rollout_sess = create_policy_agent(
        hidden_layers=[64],
        loss_str="a2c",
        name="current_rollout"
    )

    # 如果对手池为空，添加初始模型
    if len(opponent_pool.opponents) == 0:
        print("🆕 对手池为空，创建初始模型...")

        # 评估初始模型
        print("📊 评估初始模型...")
        eval_wins = 0
        for i in range(5):
            result = play_game_between_agents(deep_agent, None)
            if result > 0:
                eval_wins += 1

        initial_win_rate = eval_wins / 5

        # 添加到对手池
        opponent_pool.add_double_agent(
            deep_agent, rollout_agent,
            name=f"initial_wr{initial_win_rate:.2f}",
            win_rate=initial_win_rate
        )

        print(f"✅ 创建初始模型，胜率: {initial_win_rate:.2%}")

    iteration_stats = []

    for iteration in range(total_iterations):
        print(f"\n{'=' * 60}")
        print(f"🎯 第 {iteration + 1}/{total_iterations} 轮迭代")
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
                print(f"⚔️  对战对手: {opponent_name} (Elo: {elo:.0f})")
            else:
                opponent_name = None
        else:
            opponent_name = None

        # 2. 进行训练对局
        print(f"\n🎮 开始训练对局...")
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

        print(f"\n📊 训练结果:")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  平均结果: {avg_result:.3f}")

        # 4. 更新Elo（如果是对手池中的对手）
        if opponent_name and opp_deep_agent:
            current_agent_name = f"iter{iteration:03d}"

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
        agent_name = f"iter{iteration:03d}_wr{win_rate:.2f}"
        print(f"\n💾 保存当前模型到对手池: {agent_name}")

        opponent_pool.add_double_agent(
            deep_agent, rollout_agent,
            name=agent_name,
            win_rate=win_rate
        )

        # 6. 评估当前agent性能
        print(f"\n📈 评估当前agent性能...")
        eval_wins = 0
        eval_results = []

        for i in range(eval_games):
            result = play_game_between_agents(deep_agent, None)
            eval_results.append(result)
            if result > 0:
                eval_wins += 1

        eval_win_rate = eval_wins / eval_games
        eval_avg_result = np.mean(eval_results)

        print(f"🎯 评估结果:")
        print(f"  胜率: {eval_win_rate:.2%}")
        print(f"  平均结果: {eval_avg_result:.3f}")

        # 7. 保存检查点
        print(f"\n💾 保存检查点...")
        deep_agent.save(f"./saved_models/opponent_pool/deep_iteration_{iteration}")
        rollout_agent.save(f"./saved_models/opponent_pool/rollout_iteration_{iteration}")

        iteration_stats.append({
            'iteration': iteration,
            'train_win_rate': win_rate,
            'eval_win_rate': eval_win_rate,
            'eval_avg_result': eval_avg_result,
            'opponent': opponent_name if opponent_name else "random"
        })

    # 保存最终模型
    print("\n💾 保存最终模型...")
    deep_agent.save("./saved_models/opponent_pool/deep_policy_final")
    rollout_agent.save("./saved_models/opponent_pool/rollout_policy_final")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 训练完成统计")
    print("=" * 60)

    for stat in iteration_stats:
        print(f"Iteration {stat['iteration']:2d}: "
              f"训练胜率={stat['train_win_rate']:5.1%} | "
              f"评估胜率={stat['eval_win_rate']:5.1%} | "
              f"平均结果={stat['eval_avg_result']:6.3f} | "
              f"对手={stat['opponent']}")

    print("=" * 60)

    # 打印对手池最终状态
    opponent_pool.print_pool_status()

    # 关闭会话
    deep_sess.close()
    rollout_sess.close()

    return deep_agent, rollout_agent, iteration_stats


if __name__ == "__main__":
    # 先删除旧的对手池和模型，从头开始
    import shutil
    import os

    # 删除旧文件
    if os.path.exists("./opponent_pool"):
        shutil.rmtree("./opponent_pool")
        print("🗑️  删除旧对手池")

    if os.path.exists("./saved_models/opponent_pool"):
        shutil.rmtree("./saved_models/opponent_pool")
        print("🗑️  删除旧模型")

    os.makedirs("./saved_models", exist_ok=True)

    # 开始训练
    deep_agent, rollout_agent, stats = train_with_opponent_pool_simple()