# 人工智能导论2024-作业1: Bait游戏

## 环境安装

在当前目录运行 `pip install -r requirements.txt` 安装所需代码库。

安装成功后，可运行 `python play.py` 用键盘方向键畅玩推箱子。可通过修改第 49 行的 `level` 变量为 0~4 设置不同关卡。

熟悉游戏后，即可运行 `main.py` 开始完成作业。运行结果将存储在 `figs` 目录下，包括游戏截图与 GIF。

## main.py 运行说明

我对`main.py`进行了一点小小的修改，**运行时需要输入相应的参数来选择使用的方法和关卡**。

运行需要输入：

```cmd
python main.py --mode <运行方法> --level <关卡名>

#以下是一个例子：
python main.py --mode depthfirst --level 0
```

运行方法包括：["random", "play", "depthfirst", "limitdepthfirst", "Astar", "MCTS"]

关卡包括：['0','1','2','3','4']

也可输入 --help 查询：
  ```cmd
  python main.py --help
  ```

