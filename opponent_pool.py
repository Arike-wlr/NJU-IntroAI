import os
import pickle
import numpy as np
from agent.agent import GoPolicyAgent, create_policy_agent, load_policy_agent
import tensorflow as tf


class OpponentPool:
    def __init__(self, pool_dir="./opponent_pool", max_size=10, board_size=5):
        self.pool_dir = pool_dir
        self.max_size = max_size
        self.board_size = board_size
        self.opponents = []  # 存储模型路径（不带后缀）
        self.elo_ratings = {}  # Elo评分
        self.win_rates = {}
        self.model_types = {}  # 记录模型类型: "deep" 或 "rollout"

        os.makedirs(pool_dir, exist_ok=True)
        self.load_pool()

    def load_pool(self):
        """加载对手池"""
        if os.path.exists(f"{self.pool_dir}/pool_info.pkl"):
            with open(f"{self.pool_dir}/pool_info.pkl", "rb") as f:
                data = pickle.load(f)
                self.opponents = data.get('opponents', [])
                self.elo_ratings = data.get('elo_ratings', {})
                self.win_rates = data.get('win_rates', {})
                self.model_types = data.get('model_types', {})

    def save_pool(self):
        """保存对手池"""
        data = {
            'opponents': self.opponents,
            'elo_ratings': self.elo_ratings,
            'win_rates': self.win_rates,
            'model_types': self.model_types
        }
        with open(f"{self.pool_dir}/pool_info.pkl", "wb") as f:
            pickle.dump(data, f)

    def add_opponent(self, agent, model_type="deep", name=None, win_rate=None):
        """添加新对手到池中

        Args:
            agent: GoPolicyAgent实例
            model_type: "deep" 或 "rollout"
            name: 对手名称
            win_rate: 胜率
        """
        if name is None:
            import time
            name = f"model_{int(time.time())}"

        # 确保名称唯一
        base_name = name
        counter = 1
        while f"{base_name}_{model_type}" in self.opponents:
            base_name = f"{name}_{counter}"
            counter += 1

        name = base_name

        # 保存模型
        model_path = f"{self.pool_dir}/{name}_{model_type}"
        agent.save(model_path)

        # 添加到列表
        model_key = f"{name}_{model_type}"
        if model_key not in self.opponents:
            self.opponents.append(model_key)

        # 记录模型类型
        if name not in self.model_types:
            self.model_types[name] = {'deep': False, 'rollout': False}
        self.model_types[name][model_type] = True

        # 初始化Elo评分（默认1500）
        if name not in self.elo_ratings:
            self.elo_ratings[name] = 1500

        # 记录胜率
        if win_rate is not None:
            self.win_rates[name] = win_rate

        # 如果超过最大数量，移除最弱的对手
        if len(self.opponents) > self.max_size:
            self._remove_weakest()

        self.save_pool()
        print(f"✅ 对手 '{name}_{model_type}' 已添加到对手池")
        return name

    def add_double_agent(self, deep_agent, rollout_agent, name=None, win_rate=None):
        """添加双层网络对手"""
        if name is None:
            import time
            name = f"model_{int(time.time())}"

        # 保存深度网络
        self.add_opponent(deep_agent, model_type="deep", name=name, win_rate=win_rate)

        # 保存浅层网络
        self.add_opponent(rollout_agent, model_type="rollout", name=name, win_rate=win_rate)

        return name

    def _remove_weakest(self):
        """移除最弱的对手"""
        if not self.opponents:
            return

        # 根据Elo评分排序，移除最低分的
        opponent_names = set([name.split('_')[0] for name in self.opponents])
        opponents_with_elo = [(name, self.elo_ratings.get(name, 1500))
                              for name in opponent_names]
        opponents_with_elo.sort(key=lambda x: x[1])

        weakest = opponents_with_elo[0][0]

        # 移除所有相关模型
        to_remove = [name for name in self.opponents if name.startswith(f"{weakest}_")]
        for model_key in to_remove:
            self.opponents.remove(model_key)

            # 删除模型文件
            model_files = [f for f in os.listdir(self.pool_dir)
                           if f.startswith(model_key)]
            for file in model_files:
                os.remove(f"{self.pool_dir}/{file}")

        # 从记录中删除
        if weakest in self.model_types:
            del self.model_types[weakest]
        if weakest in self.win_rates:
            del self.win_rates[weakest]
        if weakest in self.elo_ratings:
            del self.elo_ratings[weakest]

        print(f"🗑️  移除最弱对手: {weakest}")

    def get_opponent(self, strategy="balanced", require_rollout=False):
        """根据策略选择对手"""
        if not self.opponents:
            return None

        # 获取所有基础名称（去重）
        base_names = set()
        for model_key in self.opponents:
            parts = model_key.split('_')
            base_name = '_'.join(parts[:-1])  # 去掉最后一部分（deep/rollout）
            base_names.add(base_name)

        # 过滤有浅层网络的对手（如果需要）
        candidate_names = list(base_names)
        if require_rollout:
            candidate_names = [
                name for name in base_names
                if self.model_types.get(name, {}).get('rollout', False)
            ]
            if not candidate_names:
                print("⚠️  没有找到有浅层网络的对手，返回所有对手")
                candidate_names = list(base_names)

        if not candidate_names:
            return None

        if strategy == "balanced":
            # 平衡选择：给中等水平更高权重
            weights = []
            for name in candidate_names:
                elo = self.elo_ratings.get(name, 1500)
                # 偏离平均Elo越远，权重越低
                weight = 1.0 / (1 + abs(elo - 1500) / 100)
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.random.choice(candidate_names, p=weights)

        elif strategy == "strongest":
            # 选择最强的对手
            return max(candidate_names,
                       key=lambda x: self.elo_ratings.get(x, 1500))

        elif strategy == "weakest":
            # 选择最弱的对手（用于训练初期）
            return min(candidate_names,
                       key=lambda x: self.elo_ratings.get(x, 1500))

        elif strategy == "random":
            return np.random.choice(candidate_names)

    def load_agent(self, name, model_type="deep"):
        """加载指定名称和类型的agent"""
        model_path = f"{self.pool_dir}/{name}_{model_type}"

        if not tf.train.checkpoint_exists(model_path):
            print(f"❌ 找不到模型文件: {model_path}")
            return None, None

        # 根据模型类型确定网络结构
        if model_type == "deep":
            hidden_layers = self.model_types.get(name, {}).get('hidden_layers', [256, 256])
        else:  # rollout
            hidden_layers = [64]  # 浅层网络固定配置

        # 使用工厂函数加载agent
        agent, session = load_policy_agent(
            save_path=model_path,
            hidden_layers=hidden_layers,
            loss_str="a2c",
            name=f"{name}_{model_type}"
        )

        if agent:
            print(f"✅ 加载 {name}_{model_type} 成功")
            return agent, session
        else:
            return None, None

    def load_double_agent(self, name):
        """加载双层网络agent"""
        # 加载深度网络
        deep_agent, deep_sess = self.load_agent(name, model_type="deep")
        if not deep_agent:
            return None, None, None, None

        # 加载浅层网络
        rollout_agent, rollout_sess = self.load_agent(name, model_type="rollout")

        return deep_agent, rollout_agent, deep_sess, rollout_sess

    def update_elo(self, agent1_name, agent2_name, result, k=32):
        """更新Elo评分"""
        # 确保agent在记录中
        if agent1_name not in self.elo_ratings:
            self.elo_ratings[agent1_name] = 1500
        if agent2_name not in self.elo_ratings:
            self.elo_ratings[agent2_name] = 1500

        r1 = self.elo_ratings[agent1_name]
        r2 = self.elo_ratings[agent2_name]

        # 预期胜率
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1

        # 更新评分
        self.elo_ratings[agent1_name] = r1 + k * (result - e1)
        self.elo_ratings[agent2_name] = r2 + k * ((1 - result) - e2)

        print(f"📊 Elo更新: {agent1_name}({r1:.0f}→{self.elo_ratings[agent1_name]:.0f}) "
              f"vs {agent2_name}({r2:.0f}→{self.elo_ratings[agent2_name]:.0f}) "
              f"结果={result}")

        self.save_pool()

    def get_pool_info(self):
        """获取对手池信息"""
        info = []
        base_names = set()
        for model_key in self.opponents:
            parts = model_key.split('_')
            base_name = '_'.join(parts[:-1])
            base_names.add(base_name)

        for name in sorted(base_names,
                           key=lambda x: self.elo_ratings.get(x, 1500),
                           reverse=True):
            elo = self.elo_ratings.get(name, 1500)
            win_rate = self.win_rates.get(name, 0.0)
            model_type = self.model_types.get(name, {})
            has_deep = model_type.get('deep', False)
            has_rollout = model_type.get('rollout', False)

            info.append({
                'name': name,
                'elo': elo,
                'win_rate': win_rate,
                'has_deep': has_deep,
                'has_rollout': has_rollout
            })

        return info

    def print_pool_status(self):
        """打印对手池状态"""
        print("\n" + "=" * 60)
        print("🎯 对手池状态")
        print("=" * 60)

        info = self.get_pool_info()
        for i, model_info in enumerate(info, 1):
            deep_str = "✓" if model_info['has_deep'] else "✗"
            rollout_str = "✓" if model_info['has_rollout'] else "✗"
            print(f"{i:2d}. {model_info['name']:20s} "
                  f"Elo: {model_info['elo']:5.0f} | "
                  f"胜率: {model_info['win_rate']:5.1%} | "
                  f"深度: {deep_str} | "
                  f"浅层: {rollout_str}")

        print("=" * 60)