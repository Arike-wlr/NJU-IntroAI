# 人工智能导论2024-个人作业记录

## Branch说明

`HW1`  Bait 游戏 \- 搜索 (Search)

`HW2`  黑白棋游戏 \- 博弈 (Game)

`HW3`  Aliens 游戏 \- 监督学习 (Supervised learning)

`HW4`  CartPole 游戏 \- 强化学习 (Reinforcement learning)

`HW5`  Mini Alpha Go (选做) - 强化学习和博弈

（hw1-1 为废弃分支，不用在意）

---

## 交作业方法（cmd）

```cmd
C:\Users\HUAWEI>sftp -P 22 IntroAI@www.lamda.nju.edu.cn
#在存有待交作业的目录下的cmd中输入以上内容

The authenticity of host 'www.lamda.nju.edu.cn (210.28.132.67)' can't be established.
ED25519 key fingerprint is SHA256:QzFR6J7izXd4x5u/87dZim0r0N/XAi1yVyIgzipMTpY.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
Warning: Permanently added 'www.lamda.nju.edu.cn' (ED25519) to the list of known hosts.
#首次可能会出现以上输出，不用在意（应该是这样，记不清了，或许也是要输入yes）

IntroAI@www.lamda.nju.edu.cn's password: course01234!@#$
#输入密码

Connected to www.lamda.nju.edu.cn.
#已进入，可以交作业了

sftp> cd /D:/Courses/IntroAi_HW/hw1
#进入官网指定的作业目录，注意每次指定的目录都不一样，一定别交错了

stfp> put 241300009.zip
# put 命令传入作业

stfp> ls 
# ls 命令可查看是否提交成功
```

