import os
os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from environment.go import Position
from environment import coords
from algorimths.mcts import AlphaGoMCTS,MCTS


def print_simple_board(board):
    """打印简化版5x5棋盘"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print("  0 1 2 3 4")
    for i in range(5):
        row = [symbols[board[i, j]] for j in range(5)]
        print(f"{i} {' '.join(row)}")


def test_model_loading():
    """测试模型加载"""
    # 检查模型文件
    deep_path = "../saved_models/policy_networks/deep_policy_final"
    rollout_path = "../saved_models/policy_networks/rollout_policy_final"

    print(f"模型文件检查:")
    print(f"  深度网络: {os.path.exists(deep_path + '.index')}")
    print(f"  浅层网络: {os.path.exists(rollout_path + '.index')}")

    if not os.path.exists(deep_path + ".index"):
        print("⚠️ 找不到深度网络")

    if not os.path.exists(rollout_path + ".index"):
        print("⚠️ 找不到浅层网络")

    return deep_path, rollout_path


def test_neural_networks(alphago):
    """测试神经网络预测"""
    # 创建一个测试局面
    game = Position(komi=0.5)

    print("1. 测试深度网络预测...")
    try:
        deep_probs = alphago.deep_policy.get_action_probs(game)
        print(f"   预测成功！形状: {deep_probs.shape}")

        # 分析预测结果
        best_action = np.argmax(deep_probs)
        best_prob = deep_probs[best_action]
        print(f"   最佳动作: {best_action}, 概率: {best_prob:.4f}")

        # 检查是否均匀分布
        uniform_prob = 1.0 / len(deep_probs)
        is_uniform = np.abs(best_prob - uniform_prob) < 0.01
        print(f"   是否接近均匀分布: {is_uniform}")

        if is_uniform:
            print("   ⚠️ 网络可能没有学好，或者刚初始化")
        else:
            print("   ✅ 网络有学习效果！")

    except Exception as e:
        print(f"   ❌ 深度网络预测失败: {e}")

    print("2. 测试浅层网络预测...")
    try:
        rollout_probs = alphago.rollout_policy.get_action_probs(game)
        print(f"   预测成功！形状: {rollout_probs.shape}")

        best_action = np.argmax(rollout_probs)
        best_prob = rollout_probs[best_action]
        print(f"   最佳动作: {best_action}, 概率: {best_prob:.4f}")

    except Exception as e:
        print(f"   ❌ 浅层网络预测失败: {e}")


def test_mcts_search(alphago):
    """测试MCTS搜索"""
    game = Position(komi=0.5)
    print("初始棋盘:")
    print_simple_board(game.board)

    print("执行AlphaGo MCTS搜索...")
    try:
        action_probs = alphago.search(game, current_player=0, num_simulations=20)

        print(f"搜索完成！动作概率形状: {action_probs.shape}")

        sorted_indices = np.argsort(action_probs)[-5:][::-1]
        print("最有可能的5个动作:")
        for idx in sorted_indices:
            prob = action_probs[idx]
            if idx == 25:
                print(f"  PASS: {prob:.4f}")
            else:
                point = coords.from_flat(idx)
                print(f"  ({point[0]}, {point[1]}): {prob:.4f}")

        best_action = np.argmax(action_probs)
        return best_action, action_probs

    except Exception as e:
        print(f"❌ MCTS搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_with_random_mcts(alphago):
    """对比神经网络MCTS和随机MCTS"""
    # 创建随机MCTS
    random_mcts = MCTS(exploration_weight=1.0, simulation_limit=10)

    # 测试几个不同的局面
    test_games = []
    for i in range(3):
        game = Position(komi=0.5)
        # 随机走几步，创建不同局面
        for j in range(i * 2):
            legal_moves = game.all_legal_moves()
            legal_actions = np.where(legal_moves == 1)[0]
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
        test_games.append(game)

    print(f"测试 {len(test_games)} 个不同局面")

    for i, game in enumerate(test_games):
        print(f"局面 {i + 1}:")
        print_simple_board(game.board)

        # 神经网络MCTS
        import time
        start = time.time()
        neural_action, neural_probs = alphago.get_best_action(
            game, 0, num_simulations=30
        )
        neural_time = time.time() - start

        # 随机MCTS
        start = time.time()
        random_action, random_probs = random_mcts.get_best_action(
            game, 0, num_simulations=30
        )
        random_time = time.time() - start

        print(f"  神经网络MCTS: 动作={neural_action}, 时间={neural_time:.2f}s")
        print(f"  随机MCTS:     动作={random_action}, 时间={random_time:.2f}s")

        # 简单评估：谁的动作更靠近中心
        def distance_to_center(action):
            if action == 25:  # PASS
                return 10  # 很大的值
            point = coords.from_flat(action)
            center = 2  # 5x5棋盘的中心是(2,2)
            return abs(point[0] - center) + abs(point[1] - center)

        neural_dist = distance_to_center(neural_action)
        random_dist = distance_to_center(random_action)

        if neural_dist < random_dist:
            print(f"  ✅ 神经网络动作更好（更靠近中心）")
        elif random_dist < neural_dist:
            print(f"  ⚠️  随机动作更好")
        else:
            print(f"  ➖ 两者相当")


def play_demo_game(alphago):
    """演示对弈"""
    print("对弈演示：AlphaGo MCTS vs 随机AI")

    game = Position(komi=0.5)
    step = 0
    max_steps = 10  # 演示10步

    print("初始棋盘:")
    print_simple_board(game.board)
    print("\nAlphaGo MCTS执黑（X），随机AI执白（O）")

    while not game.is_game_over() and step < max_steps:
        print(f"\n--- 第{step + 1}步 ---")

        current_player = 0 if game.to_play == 1 else 1

        if current_player == 0:  # AlphaGo MCTS
            print("AlphaGo MCTS思考中...")
            action, probs = alphago.get_best_action(
                game, current_player,
                num_simulations=50,
                temperature=0.5
            )

            if action == 25:
                print("AlphaGo: PASS")
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                print(f"AlphaGo落子: ({point[0]}, {point[1]})")
                game = game.play_move(point, mutate=False)

                # 更新MCTS根节点
                alphago.update_root(action)

        else:  # 随机AI
            legal_moves = game.all_legal_moves()
            legal_actions = np.where(legal_moves == 1)[0]

            if len(legal_actions) == 0:
                action = 25
            else:
                action = np.random.choice(legal_actions)

            if action == 25:
                print("随机AI: PASS")
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                print(f"随机AI落子: ({point[0]}, {point[1]})")
                game = game.play_move(point, mutate=False)

        # 显示当前棋盘
        print("\n当前棋盘:")
        print_simple_board(game.board)

        step += 1

    # 游戏结果
    print("\n" + "=" * 40)
    print("演示结束!")
    print(f"总步数: {step}")
    if game.is_game_over():
        print(f"最终结果: {game.result_string()}")
    else:
        print("达到最大步数限制")


def main():
    """主测试函数"""
    # 1. 测试模型加载
    deep_path, rollout_path = test_model_loading()
    if not deep_path or not rollout_path:
        print("❌ 无法加载模型，请先训练模型")
        return

    # 2. 创建AlphaGo MCTS
    print("创建AlphaGo MCTS...")
    try:
        alphago = AlphaGoMCTS(
            deep_policy_agent=deep_path,
            rollout_policy_agent=rollout_path,
            exploration_weight=1.0,
            simulation_limit=10
        )
        print("✅ AlphaGo MCTS创建成功")

    except Exception as e:
        print(f"❌ 创建AlphaGo MCTS失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 测试神经网络
    test_neural_networks(alphago)

    # 4. 测试MCTS搜索
    best_action, probs = test_mcts_search(alphago)

    # 5. 性能对比（可选，时间较长）
    user_input = input("\n是否进行性能对比测试？(y/n): ")
    if user_input.lower() == 'y':
        compare_with_random_mcts(alphago)

    # 6. 对弈演示
    user_input = input("\n是否观看对弈演示？(y/n): ")
    if user_input.lower() == 'y':
        play_demo_game(alphago)

    # 7. 清理
    print("\n清理资源...")
    alphago.deep_sess.close()
    alphago.rollout_sess.close()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()