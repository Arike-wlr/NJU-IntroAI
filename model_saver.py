import os
import time
import pickle


class ModelSaver:
    """模型保存管理器"""

    def __init__(self, save_dir="./saved_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 创建子目录
        self.policy_dir = os.path.join(save_dir, "policy_networks")
        self.value_dir = os.path.join(save_dir, "value_networks")
        os.makedirs(self.policy_dir, exist_ok=True)
        os.makedirs(self.value_dir, exist_ok=True)

    def save_policy_network(self, agent, name, iteration=None):
        """保存策略网络"""
        if iteration is not None:
            filename = f"{name}_iter_{iteration}"
        else:
            filename = name

        # 1. 保存TensorFlow模型
        tf_path = os.path.join(self.policy_dir, filename)
        agent._saver.save(agent._session, tf_path)

        # 2. 保存元数据
        metadata = {
            'name': name,
            'iteration': iteration,
            'hidden_layers': agent._layer_sizes,
            'num_actions': agent._num_actions,
            'save_time':time.time()
        }

        meta_path = os.path.join(self.policy_dir, f"{filename}_meta.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Policy network saved: {tf_path}")
        return tf_path

    def save_value_network(self, network, name, iteration=None):
        """保存价值网络"""
        # 类似保存逻辑
        pass

    def load_policy_network(self, agent, filename):
        """加载策略网络"""
        tf_path = os.path.join(self.policy_dir, filename)
        agent._saver.restore(agent._session, tf_path)

        # 加载元数据
        meta_path = os.path.join(self.policy_dir, f"{filename}_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                print(f"Loaded policy network: {metadata}")

        print(f"Policy network loaded: {tf_path}")
        return metadata

    def list_saved_models(self, model_type="policy"):
        """列出所有保存的模型"""
        if model_type == "policy":
            dir_path = self.policy_dir
        else:
            dir_path = self.value_dir

        models = []
        for f in os.listdir(dir_path):
            if f.endswith('.index'):  # TensorFlow checkpoint文件
                model_name = f.replace('.index', '')
                models.append(model_name)

        return models