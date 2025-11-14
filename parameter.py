from play import AliensEnvPygame
import time
import pickle
import os
env = AliensEnvPygame(level=0, render=False)
time.sleep(0.1)  # 给print一个执行的机会
print(f'width={env.width}')

# 输出
# Hello from the pygame community. https://www.pygame.org/contribute.html
# width=32

data_list = [
        'game_records_lvl0_2025-11-03_22-35-41'
    ]
data = []
for data_load in data_list:
    with open(os.path.join('win-log', data_load, 'data.pkl'), 'rb') as f:
        data += pickle.load(f)

print (data[0][0])

#[[['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall']],
# [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']],
# [['wall'], ['portalSlow'], [], [], [], [], ['alien'], [], [], [], [], [], ['alien'], [], [], ['alien'], [], [], [], [], [], [], [], ['alien'], [], [], [], [], [], ['alien'], [], ['wall']],
# [['wall'], ['base'], ['base'], ['base'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], ['base'], ['base'], ['base'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], [], [], [], [], ['base'], ['base'], ['base'], [], [], [], [], [], [], ['base'], ['base'], ['base'], ['base'], ['base'], ['base'], [], [], [], [], [], ['base'], ['base'], ['base'], [], [], [], ['wall']], [['wall'], [], [], [], ['base'], ['base'], ['base'], ['base'], ['base'], [], [], [], [], ['base'], ['base'], ['base'], ['base'], ['base'], ['base'], ['base'], ['base'], [], [], [], ['base'], ['base'], ['base'], ['base'], ['base'], [], [], ['wall']], [['wall'], [], [], [], ['base'], [], [], [], ['base'], [], [], [], [], ['base'], ['base'], [], [], [], [], ['base'], ['base'], [], [], [], ['base'], ['base'], ['base'], ['base'], ['base'], [], [], ['wall']], [['wall'], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['avatar'], [], [], [], [], [], [], [], [], [], [], [], [], [], ['wall']], [['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall'], ['wall']]]
