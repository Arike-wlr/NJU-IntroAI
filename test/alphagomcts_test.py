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
    # deep_path = "../saved_models/opponent_pool/deep_policy_final"
    # rollout_path = "../saved_models/opponent_pool/rollout_policy_final"

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
    """神经网络MCTS vs 随机MCTS 完整对弈比较"""
    num_games = 10
    neural_wins = 0
    random_wins = 0
    draws = 0

    for game_num in range(num_games):
        print(f"\n=== 第{game_num + 1}局 ===")

        # 初始化游戏
        game = Position(komi=0.5)

        # 创建随机MCTS
        random_mcts = MCTS(exploration_weight=1.0, simulation_limit=10)

        step = 0
        max_steps = 50  # 防止无限循环

        black_player = "神经网络MCTS"
        white_player = "随机MCTS"
        black_mcts = alphago
        white_mcts = random_mcts
        print(f"神经网络MCTS执黑(X)，随机MCTS执白(O)")

        while not game.is_game_over() and step < max_steps:
            current_player = 0 if game.to_play == 1 else 1

            if current_player == 0:
                if black_player == "神经网络MCTS":
                    action, probs = black_mcts.get_best_action(
                        game, current_player,
                        num_simulations=30,
                        temperature=0.5
                    )
                else:
                    action, probs = black_mcts.get_best_action(
                        game, current_player,
                        num_simulations=30
                    )

                # 执行动作
                if action == 25:
                    game = game.pass_move(mutate=False)
                    print(f"黑方({black_player}): PASS")
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
                    print(f"黑方({black_player})落子: ({point[0]}, {point[1]})")

                    # 如果是神经网络MCTS，更新根节点
                    if black_player == "神经网络MCTS":
                        alphago.update_root(action)

            else:  # 白棋走子
                if white_player == "神经网络MCTS":
                    # 神经网络MCTS思考
                    action, probs = white_mcts.get_best_action(
                        game, current_player,
                        num_simulations=30,
                        temperature=0.5
                    )
                else:
                    # 随机MCTS思考
                    action, probs = white_mcts.get_best_action(
                        game, current_player,
                        num_simulations=30
                    )

                # 执行动作
                if action == 25:
                    game = game.pass_move(mutate=False)
                    print(f"白方({white_player}): PASS")
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
                    print(f"白方({white_player})落子: ({point[0]}, {point[1]})")

                    # 如果是神经网络MCTS，更新根节点
                    if white_player == "神经网络MCTS":
                        alphago.update_root(action)

            step += 1
            # if step % 10 == 0:
            #     print(f"\n第{step}步后棋盘:")
            #     print_simple_board(game.board)

        print("对弈结束!")
        print(f"总步数: {step}")

        # 显示最终棋盘
        print("最终棋盘:")
        print_simple_board(game.board)

        # 计算胜负
        if game.is_game_over():
            result = game.result()
            result_str = game.result_string()
            print(f"最终结果: {result_str}")

            # 根据执子方判断谁赢
            if result > 0:  # 黑胜
                winner = black_player
                if black_player == "神经网络MCTS":
                    neural_wins += 1
                else:
                    random_wins += 1
            elif result < 0:  # 白胜
                winner = white_player
                if white_player == "神经网络MCTS":
                    neural_wins += 1
                else:
                    random_wins += 1
            else:  # 平局
                winner = "平局"
                draws += 1

            print(f"本局胜者: {winner}")
        else:
            print("达到最大步数限制，判定为平局")
            draws += 1

        print(f"当前比分 - 神经网络MCTS: {neural_wins}胜, 随机MCTS: {random_wins}胜, 平局: {draws}")

        # # 每局结束后暂停一下
        # if game_num < num_games - 1:
        #     input("\n按Enter继续下一局...")

    print("对弈统计结果")
    print(f"总对局数: {num_games}")
    print(f"神经网络MCTS胜场: {neural_wins}")
    print(f"随机MCTS胜场: {random_wins}")
    print(f"平局: {draws}")

    if neural_wins > random_wins:
        print("神经网络MCTS获胜")
    elif random_wins > neural_wins:
        print("随机MCTS获胜")
    else:
        print("双方平手")


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
    # test_neural_networks(alphago)
    # 4. 测试MCTS搜索
    # best_action, probs = test_mcts_search(alphago)
    # # 5. 性能对比（可选，时间较长）
    # user_input = input("\n是否进行性能对比测试？(y/n): ")
    # if user_input.lower() == 'y':
    #
    # # 6. 对弈演示
    # user_input = input("\n是否观看对弈演示？(y/n): ")
    # if user_input.lower() == 'y':
    #     play_demo_game(alphago)
    compare_with_random_mcts(alphago)
    # 清理
    print("清理资源...")
    alphago.deep_sess.close()
    alphago.rollout_sess.close()
    print("✅ 所有测试完成！")


if __name__ == "__main__":
    main()