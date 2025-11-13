import os
import pickle
import numpy as np
import  argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from play import AliensEnvPygame

def extract_features(observation):

    # TODO

    grid = observation
    features = []

    def cell_to_feature(cell):
        object_mapping = {
            'floor': 0,
            'wall': 1,
            'avatar': 2,
            'alien': 3,
            'bomb': 4,
            'portalSlow': 5,
            'portalFast': 6,
            'sam': 7,
            'base': 8
        }
        feature_vector = [0] * len(object_mapping)
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        return feature_vector

    for row in grid:
        for cell in row:
            cell_feature = cell_to_feature(cell)
            features.extend(cell_feature)

    return np.array(features)

def main():
    data_list = [
        'game_records_lvl0_2025-11-03_22-35-41',
        'game_records_lvl0_2025-11-03_22-43-11',
        'game_records_lvl0_2025-11-03_22-50-16',
        'game_records_lvl0_2025-11-03_22-54-11',
        'game_records_lvl0_2025-11-03_22-59-24',
        'game_records_lvl0_2025-11-03_23-08-33',
        'game_records_lvl0_2025-11-03_23-13-43'
    ]
    data = []
    for data_load in data_list:
        with open(os.path.join('win-log', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    #随机森林
    def random_forest(X,y):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)
        return clf

    #SVC支持向量机
    def svc(X,y):
        clf = SVC(
            kernel='rbf',  # 径向基核函数
            C=1.0,  # 正则化参数
            gamma='scale'
        )
        clf.fit(X, y)
        return clf

    #多层感知器神经网络
    def mlp(X,y):
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            learning_rate='adaptive',
            max_iter=500
        )
        clf.fit(X, y)
        return clf

    #梯度提升
    def xgboost(X,y):
        y_mapped = y - 1
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        clf.fit(X, y_mapped)
        return clf

    parser = argparse.ArgumentParser(description="Aliens 游戏，请选择训练模式")
    parser.add_argument(
        "--method",
        choices=["rf", "svc", "mlp", "xg"],
        required=True,
        help="训练模式:rf--随机森林；svc--支持向量机；mlp--多层感知器；xg--梯度提升。"
    )
    args = parser.parse_args()

    if args.method=="rf":
        clf=random_forest(X,y)
    elif args.method=="svc":
        clf=svc(X,y)
    elif args.method=="mlp":
        clf=mlp(X,y)
    elif args.method=="xg":
        clf=xgboost(X,y)

    env = AliensEnvPygame(level=0, render=False)
    model_folder=f'models/{args.method}_lvl{env.level}'
    os.makedirs(model_folder, exist_ok=True)

    with open(f'{model_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")

if __name__ == '__main__':
    main()
