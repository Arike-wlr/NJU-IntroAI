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
    """围棋策略梯度智能体 - 改进版，解决变量名问题"""

    def __init__(self, session, hidden_layers=[128, 128], loss_str="a2c", name="agent"):
        self.board_size = 5
        self.state_size = self.board_size * self.board_size  # 25
        self.num_actions = self.board_size * self.board_size + 1  # 26

        self._session = session
        self._hidden_layers = hidden_layers
        self._loss_str = loss_str
        self._name = name  # 用于变量作用域

        # 使用变量作用域确保一致的变量名
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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

            # 收集所有变量
            self._all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

            # 创建Saver，只保存本作用域的变量
            self._saver = tf.train.Saver(var_list=self._all_vars, max_to_keep=10)

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
        probs = self._session.run(
            self.agent._policy_probs,
            feed_dict={self.agent._info_state_ph: state_batch}
        )[0]
        return probs

    def save(self, save_path):
        """保存模型"""
        self._saver.save(self._session, save_path)
        print(f"✅ [{self._name}] 模型保存到: {save_path}")

    def restore(self, save_path):
        """加载模型"""
        try:
            self._saver.restore(self._session, save_path)
            print(f"✅ [{self._name}] 模型从 {save_path} 加载成功")
            return True
        except Exception as e:
            print(f"❌ [{self._name}] 加载模型失败: {e}")
            return False

    def get_variable_names(self):
        """获取所有变量名称（用于调试）"""
        return [v.name for v in self._all_vars]


# 工厂函数，用于正确创建agent
def create_policy_agent(hidden_layers=[256, 256], loss_str="a2c", name=None):
    """创建策略梯度agent的工厂函数"""
    if name is None:
        import time
        name = f"agent_{int(time.time())}"

    # 创建新图
    tf.reset_default_graph()
    session = tf.Session()

    # 创建agent
    agent = GoPolicyAgent(
        session=session,
        hidden_layers=hidden_layers,
        loss_str=loss_str,
        name=name
    )

    # 初始化变量
    session.run(tf.global_variables_initializer())

    return agent, session


def load_policy_agent(save_path, hidden_layers=[256, 256], loss_str="a2c", name=None):
    """加载策略梯度agent"""
    if name is None:
        # 从路径提取名称
        import os
        name = os.path.basename(save_path).replace('_deep', '').replace('_rollout', '')

    # 创建新图
    tf.reset_default_graph()
    session = tf.Session()

    # 创建agent
    agent = GoPolicyAgent(
        session=session,
        hidden_layers=hidden_layers,
        loss_str=loss_str,
        name=name
    )

    # 初始化变量
    session.run(tf.global_variables_initializer())

    # 加载模型
    if agent.restore(save_path):
        return agent, session
    else:
        session.close()
        return None, None