import random, collections
import numpy as np
import tensorflow as tf
from algorimths.policy_gradient import PolicyGradient
from environment.GoEnv import TimeStep, StepType
StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class Agent(object):
    def __init__(self):
        pass

    def step(self, timestep):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, _id):
        super().__init__()
        self.player_id = _id

    def step(self, timestep):
        cur_player = timestep.observations["current_player"]
        return StepOutput(action=random.choice(timestep.observations["legal_actions"][cur_player]), probs=1.0)


class GoPolicyAgent:
    """围棋策略梯度智能体"""
    def __init__(self, session, hidden_layers=[128, 128], loss_str="a2c"):
        self.board_size = 5
        self.state_size = self.board_size * self.board_size  # 25
        self.num_actions = self.board_size * self.board_size + 1  # 26

        # 创建PolicyGradient实例
        self.agent = PolicyGradient(
            session=session,
            player_id=0,  # 总是玩家0（黑棋）
            info_state_size=self.state_size,
            num_actions=self.num_actions,
            loss_str=loss_str,  # 使用A2C算法
            hidden_layers_sizes=hidden_layers,
            batch_size=32,
            critic_learning_rate=0.01,
            pi_learning_rate=0.001,
            entropy_cost=0.01,
            num_critic_before_pi=8
        )

        self._session = session

        # 添加保存器
        self._saver = tf.train.Saver(max_to_keep=10)

        # 添加网络参数存储（用于元数据）
        self._layer_sizes = hidden_layers
        self._num_actions = self.num_actions

    def encode_state(self, position):
        """将Position对象编码为网络输入"""
        # position.board: 1=黑, -1=白, 0=空
        board = position.board.flatten()
        # 归一化到[0, 1]
        return (board + 1) / 2.0

    def select_action(self, position, is_evaluation=False):
        """选择动作"""
        state = self.encode_state(position)
        legal_moves = position.all_legal_moves()
        legal_actions = np.where(legal_moves == 1)[0]

        # 创建 observations
        observations = {
            "info_state": [state, None],
            "legal_actions": [legal_actions, None],
            "current_player": 0
        }

        # 创建 TimeStep 对象
        time_step = TimeStep(
            observations=observations,
            rewards=[0.0, 0.0],  # 当前奖励为0
            discounts=[1.0, 1.0],  # 折扣因子
            step_type=StepType.MID  # 中间步骤
        )
        step_output = self.agent.step(time_step, is_evaluation=is_evaluation)
        return step_output.action, step_output.probs

    def get_action_probs(self, position):
        """获取所有动作的概率（用于MCTS先验）"""
        state = self.encode_state(position)

        state_batch = np.reshape(state, [1, -1])
        probs = self.session.run(
            self.agent._policy_probs,
            feed_dict={self.agent._info_state_ph: state_batch}
        )[0]
        return probs