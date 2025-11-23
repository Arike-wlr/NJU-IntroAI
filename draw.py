import matplotlib.pyplot as plt
import re

# 从文件中读取训练日志数据
with open('training_log2.txt', 'r', encoding='utf-8') as file:
    data_text = file.read()

# 解析数据
episodes = []
steps = []
losses = []
returns = []

# 使用正则表达式提取数据
pattern = r'Episode (\d+) Step (\d+): Training Loss ([\d.]+|nan), Return ([\d.]+)'
matches = re.findall(pattern, data_text)

for match in matches:
    episode = int(match[0])
    step = int(match[1])
    loss = float(match[2]) if match[2] != 'nan' else None
    return_val = float(match[3])

    episodes.append(episode)
    steps.append(step)
    losses.append(loss)
    returns.append(return_val)

# 创建图表
plt.figure(figsize=(15, 10))

# 第一张图：训练损失
plt.subplot(2, 1, 1)
# 过滤掉 nan 值
valid_episodes = [ep for ep, loss in zip(episodes, losses) if loss is not None]
valid_losses = [loss for loss in losses if loss is not None]

plt.plot(valid_episodes, valid_losses, 'b-', alpha=0.7, linewidth=1)
plt.title('Training Loss over Episodes')
plt.xlabel('Episode')
plt.ylabel('Training Loss')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标，因为损失值变化范围很大

# 第二张图：回报
plt.subplot(2, 1, 2)
plt.plot(episodes, returns, 'g-', alpha=0.7, linewidth=1)
plt.title('Return over Episodes')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.grid(True, alpha=0.3)

# 添加一些统计信息
avg_return = sum(returns) / len(returns)
max_return = max(returns)
plt.axhline(y=avg_return, color='r', linestyle='--', alpha=0.5, label=f'Average Return: {avg_return:.1f}')
plt.axhline(y=max_return, color='orange', linestyle='--', alpha=0.5, label=f'Max Return: {max_return:.1f}')
plt.legend()

plt.tight_layout()
plt.show()

