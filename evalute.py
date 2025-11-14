import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from learn import extract_features
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,classification_report
)

def set_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("✓ 中文字体设置成功")
    except:
        print("⚠ 中文字体设置失败，将使用默认字体")

def load_models(model_names):
    """加载多个模型"""
    models = []

    for name in model_names:
        try:
            model_path = f'models/{name}_lvl0/gameplay_model.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            models.append(model)
            print(f"✓ 成功加载模型: {name}")
        except Exception as e:
            print(f"✗ 加载模型 {name} 失败: {e}")

    return models

def evaluate_models(models, model_names, X_test, y_test):
    """
    对比多个模型的性能

    Parameters:
    -----------
    models : list
        模型对象列表
    model_names : list
        模型名称列表
    X_test : array-like
        测试集特征
    y_test : array-like
        测试集真实标签
    classes : list, optional
        类别名称列表（用于多分类）
    """

    results = {}
    plt.figure(figsize=(15, 10))

    # 1. 计算各项指标
    print("=" * 80)
    print("模型性能对比报告")
    print("=" * 80)

    metrics_data = []

    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n{'-' * 40}")
        print(f"模型: {name}")
        print(f"{'-' * 40}")

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        if name =='xg':
            y_pred =y_pred + 1

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 存储结果
        results[name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:")
        print(cm)

        # 详细分类报告
        print(f"\n详细分类报告:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # 2. 创建指标对比表格
    metrics_df = pd.DataFrame(metrics_data)
    print("\n" + "=" * 80)
    print("模型性能汇总")
    print("=" * 80)
    print(metrics_df.round(4))

    # 3. 可视化对比
    # 图表1: 指标对比柱状图
    plt.figure(figsize=(12, 8))
    metrics_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    kind='bar')
    plt.title('模型性能指标对比', fontsize=14)
    plt.ylabel('分数', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('模型性能指标对比.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()

    # 图表2: 混淆矩阵热力图
    for i in range(0,len(model_names)):
        plt.figure(figsize=(10, 8))
        example_cm = confusion_matrix(y_test, models[i].predict(X_test))
        sns.heatmap(example_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_names[i]} - 混淆矩阵', fontsize=12)
        plt.ylabel('真实标签', fontsize=10)
        plt.xlabel('预测标签', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{model_names[i]}_混淆矩阵.png', dpi=300, bbox_inches='tight')  # 保存图片
        plt.show()

    # 图表3: 排名对比
    plt.figure(figsize=(10, 6))
    ranked_df = metrics_df.sort_values('F1-Score', ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ranked_df)))
    bars = plt.bar(range(len(ranked_df)), ranked_df['F1-Score'], color=colors)
    plt.xticks(range(len(ranked_df)), ranked_df['Model'], rotation=45)
    plt.title('模型按F1分数排名', fontsize=12)
    plt.ylabel('F1分数', fontsize=10)

    # 在柱子上添加数值
    for bar, value in zip(bars, ranked_df['F1-Score']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('模型按F1分数排名.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()

    return results, metrics_df


# 使用示例
if __name__ == "__main__":
    # 示例数据
    set_chinese_font()
    model_names = ['rf', 'svc', 'mlp', 'xg']
    models = load_models(model_names)
    data_list=[
        'game_records_lvl2_2025-11-03_23-21-37',
        'game_records_lvl2_2025-11-05_09-24-35',
        'game_records_lvl2_2025-11-05_09-22-46',
        'game_records_lvl2_2025-11-05_11-21-10',
        'game_records_lvl3_2025-11-05_09-27-58',
        'game_records_lvl3_2025-11-05_11-31-51',
        'game_records_lvl4_2025-11-05_09-34-15',
        'game_records_lvl4_2025-11-05_09-43-46'
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
    X_test = np.array(X)
    y_test = np.array(y)

    # 调用函数
    results, metrics_df = evaluate_models(models, model_names, X_test, y_test)