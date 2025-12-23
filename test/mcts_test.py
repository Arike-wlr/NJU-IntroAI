# test_mcts_basic.py
import unittest
from algorimths.mcts import MCTSNode
from algorimths.mcts import MCTS


class TestMCTSBasic(unittest.TestCase):
    def test_mcts_initialization(self):
        """测试MCTS是否能正常初始化"""
        agent = MCTSAgent(num_rounds=100, c_puct=1.5)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.num_rounds, 100)
        self.assertEqual(agent.c_puct, 1.5)

    def test_mcts_select_move(self):
        """测试MCTS是否能正常返回落子"""
        agent = MCTSAgent(num_rounds=10)  # 减少模拟次数，快速测试
        board = GameState.new_game(9)  # 9路棋盘

        move = agent.select_move(board)
        print(f"AI选择了: {move}")
        self.assertIsNotNone(move)

        # 检查落子是否合法
        if move.is_play:
            self.assertTrue(board.is_valid_move(move.point))