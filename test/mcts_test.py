import os
import sys
os.environ['BOARD_SIZE'] = '5'
sys.path.append('.')
import numpy as np
from environment import coords
import importlib
import environment.go as go_module
importlib.reload(go_module)
from environment.go import Position
from algorimths.mcts import MCTS

def test_5x5_simple():
    """测试5x5棋盘"""
    print("=== 5x5棋盘测试 ===")
    print(f"设置的BOARD_SIZE: {os.environ.get('BOARD_SIZE')}")
    print(f"go模块中的N: {go_module.N}")
    try:
        game = Position(komi=0.5)
        print("\n棋盘大小验证:")
        print(f"棋盘形状: {game.board.shape}")
        print(f"应该是(5,5): {game.board.shape == (5, 5)}")
        legal_moves = game.all_legal_moves()
        print(f"\n合法走子mask形状: {legal_moves.shape}")
        print(f"应该是26 (5*5+1): {legal_moves.shape[0]}")
        board_positions = 5 * 5
        legal_actions = np.where(legal_moves == 1)[0]
        print(f"合法动作数量: {len(legal_actions)}")
        max_action = max(legal_actions) if len(legal_actions) > 0 else -1
        print(f"最大动作索引: {max_action}")
        print(f"应该是25(pass动作): {max_action == board_positions}")
        mcts = MCTS(simulation_limit=5)
        current_player = 0 if game.to_play == 1 else 1
        action, probs = mcts.get_best_action(
            game, current_player,
            num_simulations=5,
            temperature=1.0
        )
        print(f"\nMCTS选择的动作: {action}")
        if action == board_positions:  # pass
            print("动作: PASS")
        else:
            point = coords.from_flat(action)
            print(f"落子位置: 行{point[0]}, 列{point[1]}")
            if 0 <= point[0] < 5 and 0 <= point[1] < 5:
                print("✓ 位置在5x5棋盘范围内")
            else:
                print("✗ 位置超出5x5棋盘范围")
        print("\n棋盘显示:")
        print(game.__str__(colors=False))
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5x5_full_game():
    """5x5完整对局测试"""
    print("\n=== 5x5完整对局测试 ===")
    try:
        game = Position(komi=0.5)
        mcts = MCTS(simulation_limit=10)
        print("开始5x5对局...")
        step = 0
        max_steps = 20
        while not game.is_game_over() and step < max_steps:
            print(f"\n--- 第{step + 1}步 ---")
            print_simple_board(game.board)

            current_player = 0 if game.to_play == 1 else 1
            action, _ = mcts.get_best_action(
                game, current_player,
                num_simulations=10,
                temperature=1.0
            )
            board_size = 5
            if action == board_size * board_size:  # pass
                print(f"{'黑' if game.to_play == 1 else '白'}: PASS")
                game = game.pass_move(mutate=False)
            else:
                point = coords.from_flat(action)
                print(f"{'黑' if game.to_play == 1 else '白'}在 ({point[0]},{point[1]}) 落子")
                game = game.play_move(point, mutate=False)

            step += 1
        print(f"\n对局结束，共{step}步")
        if game.is_game_over():
            print(f"游戏结果: {game.result_string()}")
        return True
    except Exception as e:
        print(f"完整对局测试失败: {e}")
        return False

def print_simple_board(board):
    """打印简化版5x5棋盘"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    print("  0 1 2 3 4")
    for i in range(5):
        row = [symbols[board[i, j]] for j in range(5)]
        print(f"{i} {' '.join(row)}")

def test_coordinate_conversion():
    """测试坐标转换"""
    print("\n=== 坐标转换测试 ===")
    try:
        test_points = [(0, 0), (2, 2), (4, 4)]
        for point in test_points:
            flat = coords.to_flat(point)
            recovered = coords.from_flat(flat)

            print(f"点{point} -> 扁平坐标{flat} -> 恢复坐标{recovered}")
            print(f"转换正确: {point == recovered}")
        pass_flat = 5 * 5
        pass_point = coords.from_flat(pass_flat)
        print(f"\nPASS动作扁平坐标: {pass_flat}")
        print(f"PASS恢复为: {pass_point}")
        return True
    except Exception as e:
        print(f"坐标转换测试失败: {e}")
        return False

if __name__ == "__main__":
    tests = [
        test_5x5_simple,
        test_coordinate_conversion,
        test_5x5_full_game,
    ]
    all_passed = True
    for test in tests:
        print("\n" + "-" * 50)
        try:
            success = test()
            if success:
                print(f"{test.__name__}: ✓ 通过")
            else:
                print(f"{test.__name__}: ✗ 失败")
                all_passed = False
        except Exception as e:
            print(f"{test.__name__}: ✗ 异常 - {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有5x5测试通过!")
    else:
        print("✗ 部分测试失败")